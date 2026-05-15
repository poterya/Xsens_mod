#include "Copter.h"

#if MODE_NN_DETECT_ENABLED

#include <AP_HAL/AP_HAL.h>
#include <AP_InertialSensor/AP_InertialSensor.h>

#include "nn_detect_model.h"
#include "nn_detect_features.h"

#if CONFIG_HAL_BOARD == HAL_BOARD_SITL
  #include <cstdio>
  #include <cstdlib>
  #include <cstring>
#endif

/*
 * NN_DETECT flight mode.
 *
 * Behaviour
 * ---------
 * Holds horizontal position (Loiter-style) and altitude (Alt-Hold style)
 * with the loiter target frozen at the spot captured in init(). All pilot
 * stick input is deliberately ignored so the airframe stays still for the
 * vibration analyser. Exit by switching mode from the GCS.
 *
 * Detector state machine
 * ----------------------
 * WaitingForHover -> Detecting
 *   - WaitingForHover: armed + in-air + EKF |vxy| <= NN_HOVER_VXY_THRESHOLD
 *     and |vz| <= NN_HOVER_VZ_THRESHOLD held continuously for
 *     NN_HOVER_STABLE_MS. Progress STATUSTEXT once a second.
 *   - Detecting: every NN_VIBE_SAMPLE_PERIOD_MS the current per-axis
 *     vibration levels are pushed into a 50-deep ring; once the ring is
 *     full the 73-D feature vector is extracted and fed to the exported
 *     RandomForest model (NNDetectModel::predict_proba). The probability
 *     of class "DEFORMED" is EMA-smoothed and thresholded.
 * Disarm / landing returns to WaitingForHover and clears all state.
 *
 * Data source
 * -----------
 * AP_InertialSensor::get_vibration_levels() — built-in HP-filtered per-axis
 * RMS, identical to the value reported by the MAVLink VIBRATION message
 * and the VIBE log. total_vibration is sqrt(vx^2 + vy^2 + vz^2), matching
 * the Python pipeline (methods/main.py VibrationAnalyzer.get_vibration_rms).
 * get_accel_clip_count() is also watched; any new clip during detection is
 * surfaced as an OR-into-fault signal alongside the RF probability.
 *
 * SITL CSV replay
 * ---------------
 * In a SITL build the mode looks at the NN_DETECT_CSV environment variable.
 * If it points to a CSV with the header
 *   time_seconds,total_vibration,rms_x,rms_y,rms_z
 * (the exact format written by methods/main.py VibrationAnalyzer.save_log)
 * the detector pulls samples from the file instead of the IMU, skips the
 * hover-stability gate and starts inferring immediately after init(). Set
 * NN_DETECT_CSV_LOOP=1 to loop the file (useful for long-haul checks); by
 * default the detector holds the last sample once the file is exhausted.
 *
 * Inference
 * ---------
 * RandomForest with 200 trees, max_depth=12, trained by
 * methods/train_rf_detector.py on Normal_mod / Deformed_mod sessions. The
 * model is exported to a flat C++ table by methods/export_rf_to_cpp.py
 * (see nn_detect_model.{h,cpp}). The feature extractor in
 * nn_detect_features.cpp mirrors numpy / pandas exactly so the C++ and
 * Python inferences agree on identical input windows.
 */

// ----- Detector parameters -----

// Detection trips when EMA-smoothed P(DEFORMED) crosses this threshold.
// Same default as methods/realtime_rf_detector.py DEFAULT_DEFORMED_THRESHOLD.
static constexpr float NN_DEFORMED_THRESHOLD = 0.35f;

// EMA smoothing factor for the probability stream.
static constexpr float NN_PROBA_EMA_ALPHA = 0.25f;

// EMA smoothing factor for the raw vibration display values (STATUSTEXT only).
static constexpr float NN_VIBE_EMA_ALPHA = 0.15f;

// Vibration sample cadence into the ring (matches the 100 Hz training rate).
static constexpr uint32_t NN_VIBE_SAMPLE_PERIOD_MS = 10;

// STATUSTEXT cadence.
static constexpr uint32_t NN_REPORT_INTERVAL_MS = 1000;

// Hover stability gating.
static constexpr float    NN_HOVER_VXY_THRESHOLD = 0.30f;  // m/s
static constexpr float    NN_HOVER_VZ_THRESHOLD  = 0.30f;  // m/s
static constexpr uint32_t NN_HOVER_STABLE_MS     = 3000;   // continuous time required

static_assert(NNDetectFeatures::WINDOW_SIZE == 50, "ring size assumed 50");
static_assert(NNDetectModel::FEATURE_COUNT  == NNDetectFeatures::FEATURE_COUNT,
              "model and extractor feature counts disagree");


bool ModeNNDetect::init(bool ignore_checks)
{
    // Position hold target is the vehicle's current location. Pilot RC input
    // is deliberately ignored in this mode: vibration analysis requires a
    // motionless airframe, so we lock the target and never feed stick input
    // into loiter_nav. Exit by switching mode from the GCS.
    loiter_nav->clear_pilot_desired_acceleration();
    loiter_nav->init_target();

    if (!pos_control->D_is_active()) {
        pos_control->D_init_controller();
    }
    pos_control->D_set_max_speed_accel_m(get_pilot_speed_dn_ms(),
                                         get_pilot_speed_up_ms(),
                                         get_pilot_accel_D_mss());
    pos_control->D_set_correction_speed_accel_m(get_pilot_speed_dn_ms(),
                                                get_pilot_speed_up_ms(),
                                                get_pilot_accel_D_mss());

    csv_replay_open();

    reset_detection();
    _hover_stable_since_ms = 0;
    _fault_detected = false;
    _last_report_ms = 0;

    if (_csv_replay_active) {
        // CSV replay: data is fed from disk, the airframe state is irrelevant.
        _state = DetectState::Detecting;
        gcs().send_text(MAV_SEVERITY_INFO,
                        "NN_DETECT: engaged, replaying vibration CSV%s",
                        _csv_replay_loop ? " (loop)" : "");
    } else {
        _state = DetectState::WaitingForHover;
        gcs().send_text(MAV_SEVERITY_INFO,
                        "NN_DETECT: engaged, take off and hover to start detection");
    }
    return true;
}

void ModeNNDetect::exit()
{
    csv_replay_close();
}

void ModeNNDetect::run()
{
    // -------- 1) Standard hover control (Loiter-style, but RC is ignored) --------
    const float target_yaw_rate_rads = 0.0f;
    float target_climb_rate_ms = 0.0f;

    pos_control->D_set_max_speed_accel_m(get_pilot_speed_dn_ms(),
                                         get_pilot_speed_up_ms(),
                                         get_pilot_accel_D_mss());
    loiter_nav->clear_pilot_desired_acceleration();

    if (copter.ap.land_complete_maybe) {
        loiter_nav->soften_for_landing();
    }

    AltHoldModeState loiter_state = get_alt_hold_state_D_ms(target_climb_rate_ms);
    switch (loiter_state) {
    case AltHoldModeState::MotorStopped:
        attitude_control->reset_rate_controller_I_terms();
        attitude_control->reset_yaw_target_and_rate();
        pos_control->D_relax_controller(0.0f);
        loiter_nav->init_target();
        break;

    case AltHoldModeState::Landed_Ground_Idle:
        attitude_control->reset_yaw_target_and_rate();
        FALLTHROUGH;

    case AltHoldModeState::Landed_Pre_Takeoff:
        attitude_control->reset_rate_controller_I_terms_smoothly();
        loiter_nav->init_target();
        pos_control->D_relax_controller(0.0f);
        break;

    case AltHoldModeState::Takeoff:
        if (!takeoff.running()) {
            takeoff.start_m(constrain_float(g2.pilot_takeoff_alt_m, 0.0, 10.0));
        }
        target_climb_rate_ms = get_avoidance_adjusted_climbrate_ms(target_climb_rate_ms);
        takeoff.do_pilot_takeoff_ms(target_climb_rate_ms);
        loiter_nav->update();
        break;

    case AltHoldModeState::Flying:
        loiter_nav->update();
        target_climb_rate_ms = get_avoidance_adjusted_climbrate_ms(target_climb_rate_ms);
        pos_control->D_set_pos_target_from_climb_rate_ms(target_climb_rate_ms);
        break;
    }

    attitude_control->input_thrust_vector_rate_heading_rads(loiter_nav->get_thrust_vector(),
                                                            target_yaw_rate_rads, false);
    pos_control->D_update_controller();

    // -------- 2) Detector state machine --------
    const uint32_t now_ms = AP_HAL::millis();

    // Disarm or landing always drops us back to waiting and flushes state —
    // except when we are replaying a CSV: there the data source has nothing
    // to do with the airframe state, so the detector runs unconditionally.
    if (!_csv_replay_active && (!motors->armed() || copter.ap.land_complete)) {
        if (_state == DetectState::Detecting) {
            gcs().send_text(MAV_SEVERITY_INFO,
                            "NN_DETECT: landed/disarmed, detector paused");
        }
        _state = DetectState::WaitingForHover;
        _hover_stable_since_ms = 0;
        reset_detection();
        return;
    }

    if (!_csv_replay_active) {
        float vxy = 0.0f, vz = 0.0f;
        const bool hover_now = is_hover_stable(vxy, vz);

        if (_state == DetectState::WaitingForHover) {
            if (hover_now) {
                if (_hover_stable_since_ms == 0) {
                    _hover_stable_since_ms = now_ms;
                } else if (now_ms - _hover_stable_since_ms >= NN_HOVER_STABLE_MS) {
                    _state = DetectState::Detecting;
                    _fault_detected = false;
                    _last_report_ms = now_ms;
                    reset_detection();
                    gcs().send_text(MAV_SEVERITY_INFO,
                                    "NN_DETECT: hover stable, detection started");
                }
            } else {
                _hover_stable_since_ms = 0;
            }

            if (now_ms - _last_report_ms >= NN_REPORT_INTERVAL_MS) {
                _last_report_ms = now_ms;
                gcs().send_text(MAV_SEVERITY_INFO,
                                "NN_DETECT: waiting hover vxy=%.2f vz=%.2f",
                                (double)vxy, (double)vz);
            }
            return;
        }
    }

    // _state == DetectState::Detecting -------------------------------------

    // Push a fresh vibration sample at NN_VIBE_SAMPLE_PERIOD_MS cadence.
    // Source: CSV when replay is active, otherwise the IMU vibration monitor.
    if (now_ms - _last_sample_ms >= NN_VIBE_SAMPLE_PERIOD_MS) {
        _last_sample_ms = now_ms;
        Vector3f vibe;
        bool have_sample = false;
        if (_csv_replay_active) {
            have_sample = csv_replay_next(vibe);
        } else {
            vibe = copter.ins.get_vibration_levels();
            have_sample = true;
        }
        if (have_sample) {
            // Keep the EMA display value in sync with what is actually being
            // fed to the model (CSV value in replay, IMU value otherwise).
            if (!_vibe_ema_initialised) {
                _vibe_ema = vibe;
                _vibe_ema_initialised = true;
            } else {
                const float a = NN_VIBE_EMA_ALPHA;
                _vibe_ema.x = a * vibe.x + (1.0f - a) * _vibe_ema.x;
                _vibe_ema.y = a * vibe.y + (1.0f - a) * _vibe_ema.y;
                _vibe_ema.z = a * vibe.z + (1.0f - a) * _vibe_ema.z;
            }
            push_vibration_sample(vibe);
        }
    }

    float p_deformed = 0.0f;
    const bool inference_ready = run_inference(p_deformed);

    uint32_t new_clips = 0;
    if (!_csv_replay_active) {
        const uint8_t imu_idx = copter.ins.get_first_usable_accel();
        const uint32_t current_clips = copter.ins.get_accel_clip_count(imu_idx);
        new_clips = (current_clips > _clip_count_baseline)
                        ? (current_clips - _clip_count_baseline)
                        : 0;
    }

    const bool current_fault =
        inference_ready ? vibration_fault(_p_deformed_ema, new_clips) : false;

    if (now_ms - _last_report_ms >= NN_REPORT_INTERVAL_MS) {
        _last_report_ms = now_ms;
        if (!inference_ready) {
            gcs().send_text(MAV_SEVERITY_INFO,
                            "NN_DETECT: filling window (%u/%u)",
                            (unsigned)_ring_filled, (unsigned)NN_RING);
        } else {
            const MAV_SEVERITY sev = current_fault ? MAV_SEVERITY_WARNING
                                                   : MAV_SEVERITY_INFO;
            gcs().send_text(sev,
                            "NN_DETECT: %s p=%.2f vx=%.1f vy=%.1f vz=%.1f clip=%u",
                            current_fault ? "BROKEN" : "ok",
                            (double)_p_deformed_ema,
                            (double)_vibe_ema.x,
                            (double)_vibe_ema.y,
                            (double)_vibe_ema.z,
                            (unsigned)new_clips);
        }
    }

    if (current_fault && !_fault_detected) {
        _fault_detected = true;
        gcs().send_text(MAV_SEVERITY_CRITICAL,
                        "NN_DETECT: propeller fault detected");
    }
}

bool ModeNNDetect::is_hover_stable(float &out_vxy, float &out_vz) const
{
    const Vector3f vel_ned_ms = pos_control->get_vel_estimate_NED_ms();
    out_vxy = vel_ned_ms.xy().length();
    out_vz  = fabsf(vel_ned_ms.z);
    return (out_vxy <= NN_HOVER_VXY_THRESHOLD) &&
           (out_vz  <= NN_HOVER_VZ_THRESHOLD);
}

void ModeNNDetect::reset_detection()
{
    _ring_index = 0;
    _ring_filled = 0;
    _vibe_ema.zero();
    _vibe_ema_initialised = false;
    _p_deformed_ema = 0.0f;
    _p_ema_initialised = false;
    _last_sample_ms = 0;
    const uint8_t imu_idx = copter.ins.get_first_usable_accel();
    _clip_count_baseline = copter.ins.get_accel_clip_count(imu_idx);
}

void ModeNNDetect::push_vibration_sample(const Vector3f &v)
{
    const float total = v.length();
    _ring_total[_ring_index] = total;
    _ring_x[_ring_index] = v.x;
    _ring_y[_ring_index] = v.y;
    _ring_z[_ring_index] = v.z;
    _ring_index = (_ring_index + 1) % NN_RING;
    if (_ring_filled < NN_RING) {
        _ring_filled++;
    }
}

bool ModeNNDetect::run_inference(float &out_p_deformed)
{
    if (_ring_filled < NN_RING) {
        return false;
    }

    // Pack the ring into a linear window in chronological order.
    float total_lin[NN_RING], x_lin[NN_RING], y_lin[NN_RING], z_lin[NN_RING];
    for (uint16_t i = 0; i < NN_RING; i++) {
        const uint16_t src = (_ring_index + i) % NN_RING;
        total_lin[i] = _ring_total[src];
        x_lin[i]     = _ring_x[src];
        y_lin[i]     = _ring_y[src];
        z_lin[i]     = _ring_z[src];
    }

    float features[NNDetectModel::FEATURE_COUNT];
    NNDetectFeatures::extract(total_lin, x_lin, y_lin, z_lin, features);

    const float p = NNDetectModel::predict_proba(features);
    out_p_deformed = p;

    if (!_p_ema_initialised) {
        _p_deformed_ema = p;
        _p_ema_initialised = true;
    } else {
        const float a = NN_PROBA_EMA_ALPHA;
        _p_deformed_ema = a * p + (1.0f - a) * _p_deformed_ema;
    }
    return true;
}

bool ModeNNDetect::vibration_fault(float p_deformed_ema, uint32_t new_clips) const
{
    if (new_clips > 0) {
        return true;  // any accelerometer clip during detection is suspicious
    }
    return p_deformed_ema >= NN_DEFORMED_THRESHOLD;
}

// -----------------------------------------------------------------------------
// CSV replay (SITL only). Reads the file written by
// methods/main.py VibrationAnalyzer.save_log:
//   time_seconds,total_vibration,rms_x,rms_y,rms_z
// We ignore time_seconds (cadence is driven by NN_VIBE_SAMPLE_PERIOD_MS) and
// total_vibration (recomputed from rms_x/y/z to stay consistent with the
// in-flight code path).
// -----------------------------------------------------------------------------

void ModeNNDetect::csv_replay_open()
{
    _csv_fp = nullptr;
    _csv_replay_active = false;
    _csv_replay_loop = false;
    _csv_exhausted_reported = false;

#if CONFIG_HAL_BOARD == HAL_BOARD_SITL
    const char *path = getenv("NN_DETECT_CSV");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    FILE *fp = fopen(path, "r");
    if (fp == nullptr) {
        gcs().send_text(MAV_SEVERITY_WARNING,
                        "NN_DETECT: CSV open failed: %s", path);
        return;
    }
    // Skip header line if present (we expect one).
    char header[256];
    if (fgets(header, sizeof(header), fp) == nullptr) {
        fclose(fp);
        gcs().send_text(MAV_SEVERITY_WARNING,
                        "NN_DETECT: CSV empty: %s", path);
        return;
    }
    // If the "header" actually looks like data (no alpha characters) rewind.
    bool has_alpha = false;
    for (const char *p = header; *p; p++) {
        if ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z')) {
            has_alpha = true;
            break;
        }
    }
    if (!has_alpha) {
        rewind(fp);
    }

    _csv_fp = fp;
    _csv_replay_active = true;
    const char *loop_env = getenv("NN_DETECT_CSV_LOOP");
    _csv_replay_loop = (loop_env != nullptr && loop_env[0] != '\0' &&
                       loop_env[0] != '0');
    gcs().send_text(MAV_SEVERITY_INFO,
                    "NN_DETECT: CSV replay %s", path);
#endif
}

void ModeNNDetect::csv_replay_close()
{
#if CONFIG_HAL_BOARD == HAL_BOARD_SITL
    if (_csv_fp != nullptr) {
        fclose((FILE *)_csv_fp);
    }
#endif
    _csv_fp = nullptr;
    _csv_replay_active = false;
    _csv_replay_loop = false;
    _csv_exhausted_reported = false;
}

void ModeNNDetect::csv_replay_rewind()
{
#if CONFIG_HAL_BOARD == HAL_BOARD_SITL
    if (_csv_fp == nullptr) {
        return;
    }
    rewind((FILE *)_csv_fp);
    // Re-skip the header (always exists when we opened the file).
    char header[256];
    if (fgets(header, sizeof(header), (FILE *)_csv_fp) == nullptr) {
        // unexpected: file became empty after rewind — leave fp at EOF
    }
#endif
    _csv_exhausted_reported = false;
}

bool ModeNNDetect::csv_replay_next(Vector3f &out_vibe)
{
#if CONFIG_HAL_BOARD == HAL_BOARD_SITL
    if (_csv_fp == nullptr) {
        return false;
    }
    char line[256];
    while (true) {
        if (fgets(line, sizeof(line), (FILE *)_csv_fp) == nullptr) {
            // EOF.
            if (_csv_replay_loop) {
                csv_replay_rewind();
                continue;
            }
            if (!_csv_exhausted_reported) {
                _csv_exhausted_reported = true;
                gcs().send_text(MAV_SEVERITY_INFO,
                                "NN_DETECT: CSV exhausted, holding last sample");
            }
            return false;
        }
        // Skip empty lines.
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\n' || *p == '\r' || *p == '\0' || *p == '#') {
            continue;
        }
        float t = 0.0f, total = 0.0f, rx = 0.0f, ry = 0.0f, rz = 0.0f;
        const int n = sscanf(line, "%f,%f,%f,%f,%f",
                             &t, &total, &rx, &ry, &rz);
        if (n >= 5) {
            out_vibe.x = rx;
            out_vibe.y = ry;
            out_vibe.z = rz;
            return true;
        }
        // 3-column variant: rms_x,rms_y,rms_z
        if (sscanf(line, "%f,%f,%f", &rx, &ry, &rz) == 3) {
            out_vibe.x = rx;
            out_vibe.y = ry;
            out_vibe.z = rz;
            return true;
        }
        // unparseable line — skip and keep going
    }
#else
    (void)out_vibe;
    return false;
#endif
}

#endif  // MODE_NN_DETECT_ENABLED
