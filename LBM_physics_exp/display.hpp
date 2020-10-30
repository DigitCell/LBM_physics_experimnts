#ifndef DISPLAY_HPP
#define DISPLAY_HPP


#pragma once

#include <SFML/Graphics.hpp>

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui-SFML.h>
#include "math/cpVect.h"

#include <memory>
#include "iostream"
#include "vector"
#include "kernel.cuh"
#include "math/tinycolormap.hpp"
#include "iostream"


class Display
{
public:
    Display(const uint32_t _win_width ,
            const uint32_t _win_height);

    //offset mutators
    void setOffset(float x, float y)  {m_offsetX=x; m_offsetY=y;};
    void setOffset(const sf::Vector2i& off) {m_offsetX=off.x; m_offsetY=off.y;};

    sf::RenderTexture m_target;
    sf::Texture m_target_imgui;

    const uint32_t win_width = 1600;
    const uint32_t win_height = 900;

    uint32_t world_width = 1600;
    uint32_t world_height = 900;

    uint32_t tex_width;
    uint32_t tex_height;

    sf::RenderWindow m_window;
    sf::Clock deltaClock;
    float render_time;
    float render_fps;

    float solver_time;
    float solver_fps;

    sf::Font font;

    // sf::RenderTexture render_tex;
    sf::ContextSettings settings;
    sf::VertexArray m_va;

    float m_zoom, m_offsetX, m_offsetY, m_windowOffsetX, m_windowOffsetY;

    void Clear_window();
    void Frame_draw();
    bool ProcessEvents(parameter_set &params);

    void Draw();
    //void DrawLBM(attribute *domain);

    //using u8 = unsigned char;

    std::vector<sf::Color> Colors;
    std::vector<sf::Color> color_list;
    void DrawLBMTex(unsigned char* data);

    sf::Texture tex;
    sf::Sprite sprite;

    //source file
    sf::Texture SourceTexture;
    //sf::Sprite SourceSprite;

    //keyboard events

    bool keyLeftPressed=false;
    bool keyRightPressed=false;

    bool recalculateLsystem=false;

    bool StartPausePhysics=false;
    bool stepGraphPress=false;

    float thickness=2.5f;

    //Zoom and etc

    float zoom=1.50f;
    float colorZoom=725.00f;

    int current_Mode_Index=1;
    bool item_Mode_change=false;
    int deltaX=0;
    int deltaY=0;
    int deltaX2=0;

};

#endif // DISPLAY_HPP
