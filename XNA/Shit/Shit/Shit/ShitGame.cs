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
        public Camera camera { get; private set; }
        ModelManager modelManager;

        AudioEngine audioEngine;
        WaveBank waveBank;
        SoundBank soundBank;
        public Cue laser, death;
        // variables
        bool fullScreen = false;
       
        BasicEffect effect;
        Matrix worldTranslationMatrix = Matrix.Identity;
        Matrix worldRotationMatrix = Matrix.Identity;
        public Texture2D texture { get; private set; }
        public Texture2D texture_m { get; private set; }
        public Texture2D texture_c { get; private set; }
        public Texture2D crosshair { get; private set; }
        public Random random { get; private set; }

        float bulletSpeed = 10.0f;
        int bulletDelay = 300;
        int bulletCountdown = 0;

        public int points_p = 0;
        public int points_m = 0;
        public int level = 0;

        SpriteFont font;

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
            random = new Random();
            graphics.PreferredBackBufferWidth = 1280;
            graphics.PreferredBackBufferHeight = 720;
            RasterizerState rasterizer = new RasterizerState();
            rasterizer.CullMode = CullMode.None;
            graphics.GraphicsDevice.RasterizerState = rasterizer;
            graphics.ApplyChanges();

            texture = Content.Load<Texture2D>(@"Textures\aftertable");
            texture_m = Content.Load<Texture2D>(@"Textures\moravsky");
            texture_c = Content.Load<Texture2D>(@"Textures\dynamiteCrate_diffuse");
            crosshair = Content.Load<Texture2D>(@"Textures\crosshair");

            camera = new Camera(this, new Vector3(0.0f, 0.0f, 6.0f), Vector3.Zero, Vector3.Up, 0.1f);
            Components.Add(camera);

            modelManager = new ModelManager(this);
            Components.Add(modelManager);

            audioEngine = new AudioEngine(@"Content\Audio\Shit.xgs");
            waveBank = new WaveBank(audioEngine, @"Content\Audio\Wave Bank.xwb");
            soundBank = new SoundBank(audioEngine, @"Content\Audio\Sound Bank.xsb");
            laser = soundBank.GetCue("laser");
            death = soundBank.GetCue("death");

            font = Content.Load<SpriteFont>(@"Fonts\score");

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

            FireBullets(gameTime);

            base.Update(gameTime);
        }

        /// <summary>
        /// This is called when the game should draw itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.Red);
            
            effect.World = Matrix.Identity;
            effect.View = camera.ViewMatrix;
            effect.Projection = camera.ProjectionMatrix;

            foreach(EffectPass pass in effect.CurrentTechnique.Passes)
            {
                pass.Apply();
            }

            base.Draw(gameTime);

            spriteBatch.Begin();
            spriteBatch.Draw(crosshair, new Vector2((float)Window.ClientBounds.Width / 2.0f - crosshair.Width/2, (float)Window.ClientBounds.Height / 2.0f - crosshair.Height/2), Color.White);
            spriteBatch.DrawString(font, "Poziom: " + level.ToString() + "\nPunkty z TP: " + points_p.ToString() + "\nPunkty z SW: " + points_m.ToString(), new Vector2(0.0f, 0.0f), Color.Green);
            spriteBatch.End();
        }

        protected void ProcessKeys()
        {
            if (Keyboard.GetState().IsKeyDown(Keys.Escape) ||
                GamePad.GetState(PlayerIndex.One).IsButtonDown(Buttons.Start))
                this.Exit();
            if (Keyboard.GetState().IsKeyDown(Keys.R))
                ToggleFullscreen();
            if (Keyboard.GetState().IsKeyDown(Keys.W))
            {
                camera.Position += (camera.Speed * camera.Direction);
            }  
            if (Keyboard.GetState().IsKeyDown(Keys.S))
            {
                camera.Position -= (camera.Speed * camera.Direction);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.A))
            {
                camera.Position -= (camera.Speed * camera.Right);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.D))
            {
                camera.Position += (camera.Speed * camera.Right);
            }
            camera.Position += ((GamePad.GetState(PlayerIndex.One).ThumbSticks.Right.Y) * camera.Speed * camera.Direction);
            camera.Position += ((GamePad.GetState(PlayerIndex.One).ThumbSticks.Right.X) * camera.Speed * camera.Right);

            if(Keyboard.GetState().IsKeyDown(Keys.V))
            {
                this.IsFixedTimeStep = false;
            }
        }

        public void PlayCue(string cue)
        {
            soundBank.PlayCue(cue);
        }

        protected void FireBullets(GameTime gameTime)
        {
            if (bulletCountdown <= 0)
            {
                if (GamePad.GetState(PlayerIndex.One).IsButtonDown(Buttons.A))
                {
                    modelManager.AddBullet(camera.Position + new Vector3(0, -8, 0), bulletSpeed * camera.GetDirection());
                    bulletCountdown = bulletDelay;
                    PlayCue("laser");
                }
            }
            else bulletCountdown -= gameTime.ElapsedGameTime.Milliseconds;
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

        private Vector3 FlipYZAxes(Vector3 vec)
        {
            return new Vector3(vec.X, vec.Y, -vec.Z);
        }
    }
}
