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
    public class ShitGame : Microsoft.Xna.Framework.Game
    {
        // components
        GraphicsDeviceManager graphics;
        SpriteBatch spriteBatch;
        Camera camera;

        // variables
        bool fullScreen = false;
        VertexPositionTexture[] verts;
        int[] inds;
        VertexBuffer vertexBuffer;
        IndexBuffer indexBuffer;
        BasicEffect effect;
        Matrix worldTranslationMatrix = Matrix.Identity;
        Matrix worldRotationMatrix = Matrix.Identity;
        Texture2D texture;

        public ShitGame()
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

            camera = new Camera(this, new Vector3(3.0f, 2.0f, 5.0f), Vector3.Zero, Vector3.Up);
            Components.Add(camera);

            texture = Content.Load<Texture2D>(@"Textures\cargo");

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

            verts = new VertexPositionTexture[4];
            verts[0] = new VertexPositionTexture(new Vector3(-1.0f, 1.0f, 0.0f), new Vector2(0.0f, 0.0f));
            verts[1] = new VertexPositionTexture(new Vector3(1.0f, 1.0f, 0.0f), new Vector2(1.0f, 0.0f));
            verts[2] = new VertexPositionTexture(new Vector3(1.0f, -1.0f, 0.0f), new Vector2(1.0f, 1.0f));
            verts[3] = new VertexPositionTexture(new Vector3(-1.0f, -1.0f, 0.0f), new Vector2(0.0f, 1.0f));

            inds = new int[6] {0, 2, 3, 0, 1, 2};

            vertexBuffer = new VertexBuffer(GraphicsDevice, typeof(VertexPositionTexture), verts.Length, BufferUsage.None);
            vertexBuffer.SetData(verts);

            indexBuffer = new IndexBuffer(GraphicsDevice, IndexElementSize.ThirtyTwoBits, inds.Length, BufferUsage.None);
            indexBuffer.SetData(inds);

            effect = new BasicEffect(GraphicsDevice);
            effect.Texture = texture;
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
            // Allows the game to exit
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

            GraphicsDevice.SetVertexBuffer(vertexBuffer);
            
            effect.World = worldRotationMatrix * worldTranslationMatrix;
            effect.View = camera.ViewMatrix;
            effect.Projection = camera.ProjectionMatrix;

            foreach(EffectPass pass in effect.CurrentTechnique.Passes)
            {
                pass.Apply();

                GraphicsDevice.DrawUserIndexedPrimitives<VertexPositionTexture>(PrimitiveType.TriangleList, verts, 0, verts.Length, inds, 0, 2);
            }

            base.Draw(gameTime);
        }

        protected void ProcessKeys()
        {
            if (Keyboard.GetState().IsKeyDown(Keys.Escape))
                this.Exit();
            if (Keyboard.GetState().IsKeyDown(Keys.R))
                ToggleFullscreen();
            if (Keyboard.GetState().IsKeyDown(Keys.Left))
            {
                worldTranslationMatrix *= Matrix.CreateTranslation(new Vector3(-0.1f, 0.0f, 0.0f));
                worldRotationMatrix *= Matrix.CreateRotationY(0.1f);
            }  
            if (Keyboard.GetState().IsKeyDown(Keys.Right))
            {
                worldTranslationMatrix *= Matrix.CreateTranslation(new Vector3(0.1f, 0.0f, 0.0f));
                worldRotationMatrix *= Matrix.CreateRotationY(-0.1f);
            }
               
            if(Keyboard.GetState().IsKeyDown(Keys.V))
            {
                this.IsFixedTimeStep = false;
            }
        }

        protected void ToggleFullscreen()
        {
            if(fullScreen)
            {
                graphics.PreferredBackBufferWidth = 1280;
                graphics.PreferredBackBufferHeight = 720;
                fullScreen = false;
            }
            else
            {
                graphics.PreferredBackBufferWidth = 1920;
                graphics.PreferredBackBufferHeight = 1080;
                fullScreen = true;
            }
            graphics.ToggleFullScreen();
        }
    }
}
