#include "display.hpp"

Display::Display(uint32_t _win_width ,
                 uint32_t _win_height):

                win_width(_win_width),
                win_height(_win_height),
                m_window(sf::VideoMode(_win_width, _win_height), "Wood 01", sf::Style::Default),
                //   m_event_manager(m_window),
                m_zoom(1.0f),
                m_offsetX(0.0f),
                m_offsetY(0.0f),
                m_va(sf::Quads, 0)
{


        if (!SourceTexture.loadFromFile("../assets/exp_phys_03_small_500_blank.bmp"))
        {
            std::cout<<"Couldn't find file with source texture"<<std::endl;
        }

        tex_width =SourceTexture.getSize().x;
        tex_height=SourceTexture.getSize().y;
        world_width=2*tex_width;
        world_height=2*tex_height;

        m_target.create(world_width, world_height);
        settings.antialiasingLevel = 4;
        //m_window.setVerticalSyncEnabled(true);
        m_window.setFramerateLimit(60);


        m_windowOffsetX = m_window.getSize().x * 0.0f;
        m_windowOffsetY = m_window.getSize().y * 0.0f;
/*
        m_bodyBoxTexture.loadFromFile("../res/box_line2.png");
        m_bodyBoxSprite.setTexture(m_bodyBoxTexture);

        m_bodySphereTexture.loadFromFile("../res/circle_line2.png");
        m_bodySphereSprite.setTexture(m_bodySphereTexture);

        font.loadFromFile("../res/font.ttf");
*/
        sf::Text text;
        text.setFont(font);
        text.setCharacterSize(5);
        text.setFillColor(sf::Color(150, 150, 150));

        ImGui::SFML::Init(m_window);

        static ImGuiStyle* style = &ImGui::GetStyle();
        style->Alpha = 0.75f;

         tex.create(tex_width,tex_height);
         sprite.setTexture(tex);



        for (int i=0; i<500; i++)
        {
            const tinycolormap::Color color = tinycolormap::GetColor((float(i))/500.0f, tinycolormap::ColormapType::Viridis);
            float rc=color.r()*255;
            float gc=color.g()*255;
            float bc=color.b()*255;

            color_list.push_back(sf::Color(rc,gc,bc));

        }

                //m_swarm.setJob([this](std::vector<up::Body>& data, uint32_t id, uint32_t step) {updateVertexArray(data, id, step); });
}

void Display::Clear_window()
{
     m_window.clear();
     m_target.clear(sf::Color::White);
     Draw();
}

void Display::Frame_draw()
{

    m_window.display();
}

bool Display::ProcessEvents(parameter_set &params)
{
    sf::Event event;
        while (m_window.pollEvent(event)) {
           ImGui::SFML::ProcessEvent(event);

           if (event.type == sf::Event::Closed) {
               ImGui::SFML::Shutdown();
               m_window.close();
               return false;

           }

           if(sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
               keyLeftPressed=true;
           else
               keyLeftPressed=false;

           if(sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
               keyRightPressed=true;
           else
               keyRightPressed=false;
        }

        if(m_window.isOpen())
        {

            ImGui::SFML::Update(m_window, deltaClock.restart());

            ImGui::Begin("Particle experiments");
                ImGui::Text("frame rate %f", render_time+solver_time);
                ImGui::Text("frame fps %f", 1000.0f/(render_time+solver_time));

                ImGui::Text("frame graph rate %f", render_time);
                ImGui::Text("frame graph fps %f", render_fps);

                ImGui::Text("frame solver rate %f", solver_time);
                ImGui::Text("frame solver fps %f", solver_fps);


                if (ImGui::Button("Step Simulation"))
                {
                   stepGraphPress=true;

                }


                ImGui::Checkbox("Start-Pause Simulation", &StartPausePhysics);
                params.need_update+=ImGui::SliderInt("Steps in frame %f", &params.stepsPerRender, 1, 25);




                //ImGui::SliderFloat("Thikness %f", &thickness, 0.5f , 7.0f);


                 ImGui::SliderFloat("Use tree zoom %f", &zoom, 0.05f , 10.0f);
                 //ImGui::SliderFloat("Color range coeff %f", &colorZoom, 1.0f , 15000.0f);
                 ImGui::Spacing();
                 params.need_update+=ImGui::SliderInt("Rho zoom coeff %f", &params.zoom_rho, 1.0f , 500.0f);
                 params.need_update+=ImGui::SliderInt("Curlr zoom coeff %f", &params.zoom_curl, 1.0f , 1500.0f);
                 params.need_update+=ImGui::SliderInt("U zoom coeff %f", &params.zoom_u, 1.0f , 3500.0f);

                 ImGui::Spacing();
                 params.need_update+=ImGui::SliderFloat("Params Rhof %f", &params.viscosity, 0.003f , 0.15f);
                 params.need_update+=ImGui::SliderFloat("Params Velocity %f", &params.v, 0.000f , 0.45f);

                 ImGui::Spacing();
                 const char* listbox_items[] = { "RHO", "Curl", "Ux", "Uy"};
                 params.need_update+=ImGui::ListBox("Select Mode", &current_Mode_Index, listbox_items,IM_ARRAYSIZE(listbox_items), 4);


                 //ImGui::SliderInt("Use deltaX %d", &deltaX, -4000 , 4000);
                 //ImGui::SliderInt("use deltaY %d", &deltaY, -4000 , 4000);
                 //ImGui::SliderInt("SeconddeltaX %d", &deltaX2, -1000 , 1000);


            ImGui::End();

            ImGui::Begin("Particle Window");
               m_target_imgui=m_target.getTexture();
               ImGui::Image(m_target_imgui);

            ImGui::End();


            ImGui::EndFrame();
            ImGui::SFML::Render(m_window);
        }

        return true;
}

void Display::Draw()
{
    sf::Clock clock;
    sf::RectangleShape ground(sf::Vector2f(world_width, world_height));
    ground.setFillColor(sf::Color(25,25,25,255));

    sf::RenderStates rs_ground;
    rs_ground.transform.translate(m_windowOffsetX, m_windowOffsetY);
    rs_ground.transform.scale(m_zoom, m_zoom);
    rs_ground.transform.translate(-m_offsetX, -m_offsetY);

    m_target.draw(ground, rs_ground);

    sf::CircleShape circle;
    circle.setRadius(2.0f);


    sf::Transform transform;
    transform.scale(sf::Vector2f(zoom,zoom));
    transform.translate(sf::Vector2f(deltaX+world_width/2.0f,deltaY+world_height/2.0f));
    {
        circle.setPosition(-circle.getRadius(), -circle.getRadius());
        circle.setFillColor(sf::Color::White);
        m_target.draw(circle,transform);
    }
    {
        circle.setPosition(100-circle.getRadius(), -circle.getRadius());
        circle.setFillColor(sf::Color::Green);
        m_target.draw(circle,transform);
    }
    {
        circle.setPosition(-circle.getRadius(), 100 -circle.getRadius());
        circle.setFillColor(sf::Color::Blue);
        m_target.draw(circle,transform);
    }

    render_time = clock.getElapsedTime().asMicroseconds() * 0.001f;
    render_fps = 1000.0f/render_time;

}



void Display::DrawLBMTex(unsigned char* data)
{
    tex.update(data);
    sf::Transform transform;
    transform.scale(sf::Vector2f(zoom,zoom));
    m_target.draw(sprite,transform);
}




