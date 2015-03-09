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

namespace ModelTest
{
    /// <summary>
    /// This is the main type for your game
    /// </summary>
    public class ModelTestGame : Microsoft.Xna.Framework.Game
    {
        GraphicsDeviceManager graphics;
        SpriteBatch spriteBatch;
        public Camera camera { get; private set; }
        ModelManager modelManager;
        BasicEffect effect;

        float cameraSpeed = 0.01f;

        public ModelTestGame()
        {
            graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";
        }

        /// <summary>
        /// Allows the game to perform any initialization it needs to before starting to run.
        /// This is where it can query for any required services and load any non-graphic
        /// related content.  Calling base.Initialize will enumerate through any components
        /// and initialize them as well.
        /// </summary>
        protected override void Initialize()
        {
            graphics.PreferredBackBufferWidth = 1280;
            graphics.PreferredBackBufferHeight = 720;
            RasterizerState rasterizer = new RasterizerState();
            rasterizer.CullMode = CullMode.None;
            graphics.GraphicsDevice.RasterizerState = rasterizer;
            graphics.ApplyChanges();

            camera = new Camera(this, new Vector3(0.0f, 25.0f, 100.0f), Vector3.Zero, Vector3.Up, 0.1f);
            Components.Add(camera);

            modelManager = new ModelManager(this);
            Components.Add(modelManager);

            base.Initialize();
        }

        /// <summary>
        /// LoadContent will be called once per game and is the place to load
        /// all of your content.
        /// </summary>
        protected override void LoadContent()
        {
            spriteBatch = new SpriteBatch(GraphicsDevice);

            effect = new BasicEffect(GraphicsDevice);
            effect.Texture = modelManager.texture;
            effect.TextureEnabled = true;

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
            ProcessKeys();

            base.Update(gameTime);
        }

        /// <summary>
        /// This is called when the game should draw itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.CornflowerBlue);

            effect.World = Matrix.Identity;
            effect.View = camera.ViewMatrix;
            effect.Projection = camera.ProjectionMatrix;

            foreach (EffectPass pass in effect.CurrentTechnique.Passes)
            {
                pass.Apply();
            }

            base.Draw(gameTime);
        }

        private void ProcessKeys()
        {
            if (Keyboard.GetState().IsKeyDown(Keys.Escape))
                this.Exit();
            if (Keyboard.GetState().IsKeyDown(Keys.Up))
            {
                camera.Position += (camera.Speed * camera.Direction);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.Down))
            {
                camera.Position -= (camera.Speed * camera.Direction);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.Left))
            {
                camera.Position -= (camera.Speed * camera.Right);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.Right))
            {
                camera.Position += (camera.Speed * camera.Right);
            }
        }
    }
}
