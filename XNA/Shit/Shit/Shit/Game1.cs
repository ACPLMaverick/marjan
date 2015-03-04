using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.GamerServices;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Media;

namespace Shit
{
    /// <summary>
    /// This is the main type for your game
    /// </summary>
    public class Game1 : Microsoft.Xna.Framework.Game
    {
        GraphicsDeviceManager graphics;
        SpriteBatch spriteBatch;
        Scene scene;
        Texture2D texPalette;
        Texture2D texBrick;
        Texture2D texBall;

        GameObject ball;
        GameObject palette;

        public readonly int windowWidth, windowHeight;

        const float paletteSpeed = 0.3f;
        const float ballSpeed = 0.4f;

        public Game1()
        {
            graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";

            windowWidth = Window.ClientBounds.Width;
            windowHeight = Window.ClientBounds.Height;
        }

        /// <summary>
        /// Allows the game to perform any initialization it needs to before starting to run.
        /// This is where it can query for any required services and load any non-graphic
        /// related content.  Calling base.Initialize will enumerate through any components
        /// and initialize them as well.
        /// </summary>
        protected override void Initialize()
        {
            scene = new Scene();

            base.Initialize();
        }

        /// <summary>
        /// LoadContent will be called once per game and is the place to load
        /// all of your content.
        /// </summary>
        protected override void LoadContent()
        {
            // Create a new SpriteBatch, which can be used to draw textures.
            spriteBatch = new SpriteBatch(GraphicsDevice);

            // TODO: use this.Content to load your game content here
            texPalette =  Content.Load<Texture2D>(@"Textures\tank_enemy_FR_01");
            texBall = Content.Load<Texture2D>(@"Textures\redball");
            texBrick = Content.Load<Texture2D>(@"Textures\brick");
            palette = scene.Add(new GameObject(this, "Palette", texPalette, new Vector2((windowWidth/2) - (texPalette.Width/2), windowHeight - (windowHeight/3)), 0.0f, 1.0f, Color.White));
            ball = scene.Add(new GameObject(this, "Ball", texBall, new Vector2((windowWidth / 2) - (texBall.Width*0.2f / 2), (windowHeight / 2) - (texBall.Height*0.2f / 2)), 0.0f, 0.2f, Color.White));
            scene.Add(new GameObject(this, "Brick", texBrick, new Vector2((windowWidth / 2) - (texBrick.Width*2.0f / 2), (windowHeight / 5)), 0.0f, 2.0f, Color.Green));
        }

        /// <summary>
        /// UnloadContent will be called once per game and is the place to unload
        /// all content.
        /// </summary>
        protected override void UnloadContent()
        {
            // TODO: Unload any non ContentManager content here
        }

        /// <summary>
        /// Allows the game to run logic such as updating the world,
        /// checking for collisions, gathering input, and playing audio.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Update(GameTime gameTime)
        {
            // Allows the game to exit
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed)
                this.Exit();

            ProcessKeyboard();

            scene.UpdateAll(gameTime);

            base.Update(gameTime);
        }

        /// <summary>
        /// This is called when the game should draw itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.BlanchedAlmond);

            scene.DrawAll(spriteBatch, gameTime);

            base.Draw(gameTime);
        }

        private void ProcessKeyboard()
        {
            palette.speed.X = 0.0f;
            if(Keyboard.GetState().IsKeyDown(Keys.Left))
            {
                palette.speed.X = -paletteSpeed;
            }
            if (Keyboard.GetState().IsKeyDown(Keys.Right))
            {
                palette.speed.X = paletteSpeed;
            }
            if (Keyboard.GetState().IsKeyDown(Keys.Up))
            {
                //TODO
            }
            if (Keyboard.GetState().IsKeyDown(Keys.Escape))
            {
                this.Exit();
            }

            if (palette.speed != Vector2.Zero) Vector2.Normalize(ref palette.speed, out palette.speed);
        }
    }
}
