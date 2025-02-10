#pragma once

#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <iostream>

struct window_base;

struct graphics_context
{
    window_base *window;

    graphics_context(window_base *window)
        : window(window)
    {
    }

    virtual void attach();
    virtual void detach();
    virtual void clear();
    virtual void swap_buffer();
    virtual ~graphics_context();
};

struct window_base
{
public:
    void *handle;
    std::string name;
    std::size_t width;
    std::size_t height;

    window_base(std::string name, std::size_t width, std::size_t height);

    void *get_handle() const;

    virtual bool is_closed() const;

    virtual void on_close();
    virtual void on_key(int key, int scancode, int action, int mods);
    virtual void on_char(unsigned int codepoint);
    virtual void on_scroll(double x, double y);
    virtual void on_mouse_click(int button, int action, int mods);
    virtual void on_mouse(double x, double y);
    virtual void on_enter(int entered);
    virtual void on_resize(int width, int height);
    virtual void show();
    virtual void create();
    virtual void initialize();
    virtual void finalize();
    virtual void destroy();
    virtual void update();
    virtual ~window_base() = default;

    graphics_context create_graphics_context();
};

class window_manager
{
    std::thread::id thread_id;
    std::atomic_bool should_close_flag;

    struct action_func_base
    {
        virtual void invoke(){};
        virtual ~action_func_base() = default;
    };

    template <typename Func>
    struct action_func : public action_func_base
    {
        Func func;
        action_func(Func &&_func) : func(std::move(_func)) {}
        action_func(action_func &&other) : func(std::move(other.func)) {}
        action_func &operator=(action_func &&other)
        {
            func = std::move(other.func);
            return *this;
        }
        virtual void invoke()
        {
            func();
        };
        virtual ~action_func() = default;
    };

    std::mutex mtx;
    std::deque<std::unique_ptr<action_func_base>> queue;

public:
    window_manager();

    bool should_close();
    void handle_event();
    void *create_window_handle(std::string name, int width, int height, window_base *window);
    void destroy_window_handle(window_base *window);
    void show_window(window_base *window);
    void hide_window(window_base *window);
    void initialize();
    void exit();
    void terminate();

    void get_window_size(window_base *window, int *width, int *height);
    void get_window_frame_size(window_base *window, int* left, int* top, int* right, int* bottom);

    ~window_manager() = default;

    static std::shared_ptr<window_manager> get_instance();
};

struct rendering_thread
{
    std::unique_ptr<std::thread> th;
    std::atomic_bool running;

public:
    rendering_thread() : running(false)
    {
    }

    void start(window_base *window)
    {
        running = true;
        th = std::make_unique<std::thread>([window, this]()
                                           {
            window->create();
            auto graphics_ctx = window->create_graphics_context();
            graphics_ctx.attach();
            window->initialize();
            window->show();
            while (running)
            {
                graphics_ctx.clear();
                window->update();
                graphics_ctx.swap_buffer();
            }
            window->finalize();
            graphics_ctx.detach();
            window->destroy(); });
    }

    void stop()
    {
        running = false;
        if (th && th->joinable())
        {
            th->join();
        }
    }
};

struct mouse_state
{
    double x, y;
    int right_button;
    int middle_button;
    int left_button;

    static mouse_state get_mouse_state(void *handle);
};
