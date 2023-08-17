#include "views.hpp"

void azimuth_elevation::update(mouse_state mouse)
{
    auto mouse_x = static_cast<int>(mouse.x);
    auto mouse_y = static_cast<int>(mouse.y);

    if (!on_target(mouse_x, mouse_y))
    {
        return;
    }

    if (mouse.right_button == GLFW_PRESS && on_target(mouse_x, mouse_y))
    {
        if (previous_state.right_button == GLFW_RELEASE)
        {
            begin_rotation(mouse_x, mouse_y);
        }
        else
        {
            update_rotation(mouse_x, mouse_y);
        }
    }
    else if (mouse.right_button == GLFW_RELEASE)
    {
        end_rotation();
    }

    if (mouse.middle_button == GLFW_PRESS && on_target(mouse_x, mouse_y))
    {
        if (previous_state.middle_button == GLFW_RELEASE)
        {
            begin_transition(mouse_x, mouse_y);
        }
        else
        {
            update_transition(mouse_x, mouse_y, false);
        }
    }
    else
    // else if (mouse.get_middle_button() == BUTTON_STATE::RELEASED && mouse.get_left_button() == BUTTON_STATE::RELEASED)
    {
        end_transition();
    }
    previous_state = mouse;
}
