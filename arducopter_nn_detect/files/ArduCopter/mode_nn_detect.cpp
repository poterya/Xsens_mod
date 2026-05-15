#include "Copter.h"

#if MODE_NN_DETECT_ENABLED

#include <AP_HAL/AP_HAL.h>
#include <AP_InertialSensor/AP_InertialSensor.h>

/*
 * NN_DETECT flight mode.
 *
 * Behaviour:
 *   - Holds horizontal position (like Loiter) and altitude (Alt-Hold style)
 *     so the airframe stays put while the analyser runs.
 *   - State machine for the detector itself:
 *       WaitingForHover -> Detecting
 *     The detector engages ONLY when the copter is armed, in the air and the
 *     EKF velocity estimate stays below NN_HOVER_VXY_THRESHOLD (horizontal)
 *     and NN_HOVER_VZ_THRESHOLD (vertical) continuously for
 *     NN_HOVER_STABLE_MS. Disarm / landing returns to WaitingForHover and
 *     flushes the rolling buffer.
 *   - In Detecting state every iteration samples body-frame accelerometer
 *     into a circular buffer of NN_WINDOW samples, computes the de-meaned
 *     RMS of each axis, combines them into a total vibration metric and
 *     calls vibration_fault() which is the integration hook for the trained
 *     RandomForest / DecisionTree model.
 *   - Status is reported via STATUSTEXT at NN_REPORT_INTERVAL_MS cadence,
 *     and a CRITICAL message is latched on the first fault transition.
 *
 * Inference hook:
 *   vibration_fault() currently implements a simple RMS threshold so the
 *   pipeline is fully testable end-to-end in SITL. The intent is to replace
 *   the body of this function with the auto-generated C++ inference of the
 *   sklearn model (see methods/export_model_to_cpp.py — to be added).
 */

// RMS vibration value (m/s^2) above which a propeller fault is flagged.
// MVP heuristic, will be replaced by exported sklearn model inference.
static constexpr float NN_VIB_THRESHOLD = 0.6f;

// STATUSTEXT cadence
static constexpr uint32_t NN_REPORT_INTERVAL_MS = 1000;

// hover stability gating
static constexpr float    NN_HOVER_VXY_THRESHOLD = 0.30f;  // m/s
static constexpr float    NN_HOVER_VZ_THRESHOLD  = 0.30f;  // m/s
static constexpr uint32_t NN_HOVER_STABLE_MS     = 3000;   // continuous time required


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

    reset_buffer();
    _state = DetectState::WaitingForHover;
    _hover_stable_since_ms = 0;
    _fault_detected = false;
    _last_report_ms = 0;

    gcs().send_text(MAV_SEVERITY_INFO,
                    "NN_DETECT: engaged, take off and hover to start detection");
    return true;
}

void ModeNNDetect::run()
{
    // -------- 1) Standard hover control (Loiter-style, but RC is ignored) --------
    // Targets are hard-coded to zero (no climb, no yaw, no lean from stick).
    const float target_yaw_rate_rads = 0.0f;
    float target_climb_rate_ms = 0.0f;

    pos_control->D_set_max_speed_accel_m(get_pilot_speed_dn_ms(),
                                         get_pilot_speed_up_ms(),
                                         get_pilot_accel_D_mss());

    // Explicitly clear any latent pilot acceleration request so the position
    // target stays locked to the spot we captured in init().
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

    // Disarm or landing always drops us back to waiting and flushes the buffer.
    if (!motors->armed() || copter.ap.land_complete) {
        if (_state == DetectState::Detecting) {
            gcs().send_text(MAV_SEVERITY_INFO,
                            "NN_DETECT: landed/disarmed, detector paused");
        }
        _state = DetectState::WaitingForHover;
        _hover_stable_since_ms = 0;
        reset_buffer();
        return;
    }

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
                reset_buffer();
                gcs().send_text(MAV_SEVERITY_INFO,
                                "NN_DETECT: hover stable, detection started");
            }
        } else {
            _hover_stable_since_ms = 0;
        }

        // periodic progress message so the operator sees what is happening
        if (now_ms - _last_report_ms >= NN_REPORT_INTERVAL_MS) {
            _last_report_ms = now_ms;
            gcs().send_text(MAV_SEVERITY_INFO,
                            "NN_DETECT: waiting hover vxy=%.2f vz=%.2f",
                            (double)vxy, (double)vz);
        }
        return;
    }

    // _state == DetectState::Detecting
    update_vibration_window();
    if (_samples_filled < NN_WINDOW) {
        return;
    }

    const float vib_total = compute_vibration_rms();
    const bool current_fault = vibration_fault(vib_total);

    if (now_ms - _last_report_ms >= NN_REPORT_INTERVAL_MS) {
        _last_report_ms = now_ms;
        if (current_fault) {
            gcs().send_text(MAV_SEVERITY_WARNING,
                            "NN_DETECT: BROKEN vib=%.3f", (double)vib_total);
        } else {
            gcs().send_text(MAV_SEVERITY_INFO,
                            "NN_DETECT: ok vib=%.3f", (double)vib_total);
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

void ModeNNDetect::reset_buffer()
{
    _window_index = 0;
    _samples_filled = 0;
}

void ModeNNDetect::update_vibration_window()
{
    const Vector3f &accel = copter.ins.get_accel();
    _acc_x[_window_index] = accel.x;
    _acc_y[_window_index] = accel.y;
    _acc_z[_window_index] = accel.z;
    _window_index = (_window_index + 1) % NN_WINDOW;
    if (_samples_filled < NN_WINDOW) {
        _samples_filled++;
    }
}

float ModeNNDetect::compute_vibration_rms() const
{
    float mean_x = 0.0f, mean_y = 0.0f, mean_z = 0.0f;
    for (uint16_t i = 0; i < NN_WINDOW; i++) {
        mean_x += _acc_x[i];
        mean_y += _acc_y[i];
        mean_z += _acc_z[i];
    }
    mean_x /= NN_WINDOW;
    mean_y /= NN_WINDOW;
    mean_z /= NN_WINDOW;

    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    for (uint16_t i = 0; i < NN_WINDOW; i++) {
        const float dx = _acc_x[i] - mean_x;
        const float dy = _acc_y[i] - mean_y;
        const float dz = _acc_z[i] - mean_z;
        sx += dx * dx;
        sy += dy * dy;
        sz += dz * dz;
    }
    const float rms_x = sqrtf(sx / NN_WINDOW);
    const float rms_y = sqrtf(sy / NN_WINDOW);
    const float rms_z = sqrtf(sz / NN_WINDOW);
    return sqrtf(rms_x * rms_x + rms_y * rms_y + rms_z * rms_z);
}

bool ModeNNDetect::vibration_fault(float vib_total) const
{
    // Replace this body with auto-generated sklearn inference once the model
    // is exported (see methods/export_model_to_cpp.py).
    return vib_total >= NN_VIB_THRESHOLD;
}

#endif  // MODE_NN_DETECT_ENABLED
