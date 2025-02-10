#include <cstdlib>
#include <iostream>
#include <glad/glad.h>
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <atomic>
#include <thread>
#include <future>
#include <fstream>

#include "viewer.hpp"

static constexpr int DEFAULT_SCREEN_WIDTH = 1680;
static constexpr int DEFAULT_SCREEN_HEIGHT = 1050;

bool window_manager::should_close()
{
    return should_close_flag.load();
}

window_manager::window_manager() : should_close_flag(false) {}

void window_manager::handle_event()
{
    if (std::this_thread::get_id() != thread_id)
    {
        throw std::runtime_error("Invalid call");
    }

    glfwPollEvents();

    std::vector<std::unique_ptr<action_func_base>> events;
    {
        std::lock_guard<std::mutex> lock(mtx);
        while (queue.size() > 0)
        {
            events.emplace_back(std::move(queue.front()));
            queue.pop_front();
        }
    }

    for (const auto &event : events)
    {
        event->invoke();
    }
}

static void mouse_scroll_callback(GLFWwindow *handle, double x, double y)
{
    const auto callback = static_cast<window_base *>(glfwGetWindowUserPointer((GLFWwindow *)handle));
    if (callback)
    {
        callback->on_scroll(x, y);
    }
}
static void mouse_button_callback(GLFWwindow *handle, int button, int action, int mods)
{
    const auto callback = static_cast<window_base *>(glfwGetWindowUserPointer((GLFWwindow *)handle));
    if (callback)
    {
        callback->on_mouse_click(button, action, mods);
    }
}
void mouse_cursor_callback(GLFWwindow *handle, double x, double y)
{
    const auto callback = static_cast<window_base *>(glfwGetWindowUserPointer((GLFWwindow *)handle));
    if (callback)
    {
        callback->on_mouse(x, y);
    }
}
void mouse_cursor_enter_callback(GLFWwindow *handle, int entered)
{
    const auto callback = static_cast<window_base *>(glfwGetWindowUserPointer((GLFWwindow *)handle));
    if (callback)
    {
        callback->on_enter(entered);
    }
}
void key_callback(GLFWwindow *handle, int key, int scancode, int action, int mods)
{
    const auto callback = static_cast<window_base *>(glfwGetWindowUserPointer((GLFWwindow *)handle));
    if (callback)
    {
        callback->on_key(key, scancode, action, mods);
    }
}
void char_callback(GLFWwindow *handle, unsigned int codepoint)
{
    const auto callback = static_cast<window_base *>(glfwGetWindowUserPointer((GLFWwindow *)handle));
    if (callback)
    {
        callback->on_char(codepoint);
    }
}
void window_size_callback(GLFWwindow *handle, int width, int height)
{
    const auto callback = static_cast<window_base *>(glfwGetWindowUserPointer((GLFWwindow *)handle));
    if (callback)
    {
        callback->on_resize(width, height);
    }
}
void window_close_callback(GLFWwindow *handle)
{
    const auto callback = static_cast<window_base *>(glfwGetWindowUserPointer((GLFWwindow *)handle));
    if (callback)
    {
        callback->on_close();
    }
}

void *window_manager::create_window_handle(std::string name, int width, int height, window_base *window)
{
    std::promise<GLFWwindow *> p;
    auto f = p.get_future();
    auto func = [_p = std::move(p), name, width, height, window]() mutable
    {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        auto handle = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);
        glfwSetWindowUserPointer(handle, window);
        glfwSetScrollCallback(handle, mouse_scroll_callback);
        glfwSetMouseButtonCallback(handle, mouse_button_callback);
        glfwSetCursorPosCallback(handle, mouse_cursor_callback);
        glfwSetCursorEnterCallback(handle, mouse_cursor_enter_callback);
        glfwSetKeyCallback(handle, key_callback);
        glfwSetCharCallback(handle, char_callback);
        glfwSetWindowSizeCallback(handle, window_size_callback);
        glfwSetWindowCloseCallback(handle, window_close_callback);
        _p.set_value(handle);
    };
    if (std::this_thread::get_id() == thread_id)
    {
        func();
    }
    else
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.emplace_back(std::make_unique<action_func<decltype(func)>>(std::move(func)));
        }
        glfwPostEmptyEvent();
    }
    return f.get();
}

void window_manager::destroy_window_handle(window_base *window)
{
    const auto handle = (GLFWwindow *)window->get_handle();
    auto func = [handle]() mutable
    {
        glfwDestroyWindow((GLFWwindow *)handle);
    };
    if (std::this_thread::get_id() == thread_id)
    {
        func();
    }
    else
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.emplace_back(std::make_unique<action_func<decltype(func)>>(std::move(func)));
        }
        glfwPostEmptyEvent();
    }
}

void window_manager::show_window(window_base *window)
{
    const auto handle = (GLFWwindow *)window->get_handle();
    auto func = [handle]() mutable
    {
        glfwShowWindow((GLFWwindow *)handle);
    };
    if (std::this_thread::get_id() == thread_id)
    {
        func();
    }
    else
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.emplace_back(std::make_unique<action_func<decltype(func)>>(std::move(func)));
        }
        glfwPostEmptyEvent();
    }
}

void window_manager::hide_window(window_base *window)
{
    const auto handle = (GLFWwindow *)window->get_handle();
    auto func = [handle]() mutable
    {
        glfwHideWindow((GLFWwindow *)handle);
    };
    if (std::this_thread::get_id() == thread_id)
    {
        func();
    }
    else
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.emplace_back(std::make_unique<action_func<decltype(func)>>(std::move(func)));
        }
        glfwPostEmptyEvent();
    }
}

void window_manager::initialize()
{
    thread_id = std::this_thread::get_id();

    if (glfwInit() == GL_FALSE)
    {
        throw std::runtime_error("Can't initialize GLFW");
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
}

void window_manager::exit()
{
    should_close_flag.store(true);
    glfwPostEmptyEvent();
}

void window_manager::terminate()
{
    glfwTerminate();
}

void window_manager::get_window_size(window_base *window, int *width, int *height)
{
    std::promise<std::tuple<int, int>> p;
    auto f = p.get_future();
    const auto handle = (GLFWwindow*)window->get_handle();
    auto func = [_p = std::move(p), handle]() mutable
    {
        int w, h;
        glfwGetWindowSize(handle, &w, &h);
        _p.set_value(std::make_tuple(w, h));
    };
    if (std::this_thread::get_id() == thread_id)
    {
        func();
    }
    else
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.emplace_back(std::make_unique<action_func<decltype(func)>>(std::move(func)));
        }
        glfwPostEmptyEvent();
    }
    const auto [w, h] = f.get();
    *width = w;
    *height = h;
}

void window_manager::get_window_frame_size(window_base *window, int* left, int* top, int* right, int* bottom)
{
    std::promise<std::tuple<int, int, int , int>> p;
    auto f = p.get_future();
    const auto handle = (GLFWwindow *)window->get_handle();
    auto func = [_p = std::move(p), handle]() mutable
    {
        int l, t, r, b;
        glfwGetWindowFrameSize(handle, &l, &t, &r, &b);
        _p.set_value(std::make_tuple(l, t, r, b));
    };
    if (std::this_thread::get_id() == thread_id)
    {
        func();
    }
    else
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.emplace_back(std::make_unique<action_func<decltype(func)>>(std::move(func)));
        }
        glfwPostEmptyEvent();
    }
    const auto [l, t, r, b] = f.get();
    *left = l;
    *top = t;
    *right = r;
    *bottom = b;
}

std::shared_ptr<window_manager> window_manager::get_instance()
{
    static const auto win_mgr = std::make_shared<window_manager>();
    return win_mgr;
}

window_base::window_base(std::string name, std::size_t width, std::size_t height)
    : handle(nullptr), name(name), width(width), height(height)
{
}

void *window_base::get_handle() const
{
    return handle;
}

bool window_base::is_closed() const
{
    if (handle == nullptr)
    {
        return true;
    }
    return glfwWindowShouldClose((GLFWwindow *)handle) == GL_TRUE;
}

void window_base::on_close()
{
}
void window_base::on_key(int key, int scancode, int action, int mods)
{
}
void window_base::on_char(unsigned int codepoint)
{
}
void window_base::on_scroll(double x, double y)
{
}
void window_base::on_mouse_click(int button, int action, int mods)
{
}
void window_base::on_mouse(double x, double y)
{
}
void window_base::on_enter(int entered)
{
}
void window_base::on_resize(int width, int height)
{
    this->width = width;
    this->height = height;
}
void window_base::show()
{
    window_manager::get_instance()->show_window(this);
}
void window_base::create()
{
    handle = (GLFWwindow *)window_manager::get_instance()->create_window_handle(name, DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT, this);
    if (handle == nullptr)
    {
        std::cerr << "Can't create GLFW window." << std::endl;
        return;
    }
}
void window_base::initialize()
{
}
void window_base::finalize()
{
}
void window_base::destroy()
{
    if (this->handle)
    {
        window_manager::get_instance()->destroy_window_handle(this);
    }
    this->handle = nullptr;
}
graphics_context window_base::create_graphics_context()
{
    return graphics_context(this);
}

void window_base::update() {}

mouse_state mouse_state::get_mouse_state(void *handle)
{
    mouse_state mouse;
    glfwGetCursorPos((GLFWwindow *)handle, &mouse.x, &mouse.y);
    mouse.right_button = glfwGetMouseButton((GLFWwindow *)handle, GLFW_MOUSE_BUTTON_RIGHT);
    mouse.left_button = glfwGetMouseButton((GLFWwindow *)handle, GLFW_MOUSE_BUTTON_LEFT);
    mouse.middle_button = glfwGetMouseButton((GLFWwindow *)handle, GLFW_MOUSE_BUTTON_MIDDLE);
    return mouse;
}

void graphics_context::attach()
{
    glfwMakeContextCurrent((GLFWwindow *)window->get_handle());
}
void graphics_context::detach()
{
    glfwMakeContextCurrent(nullptr);
}

void graphics_context::clear()
{
    if (window == nullptr || window->get_handle() == nullptr)
    {
        return;
    }
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void graphics_context::swap_buffer()
{
    if (window == nullptr || window->get_handle() == nullptr)
    {
        return;
    }
    glfwSwapBuffers((GLFWwindow *)window->get_handle());
}

graphics_context::~graphics_context()
{
}
