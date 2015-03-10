using System;
using System.Collections.Generic;
using System.Diagnostics;
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
    /// This is a game component that implements IUpdateable.
    /// </summary>
    public class ModelManager : Microsoft.Xna.Framework.DrawableGameComponent
    {
        private List<BasicModel> models;
        private Model plane;
        private Model crate;
        public Texture2D texture { get; private set; }

        float speed = 0.5f;

        public ModelManager(Game game)
            : base(game)
        {

        }

        /// <summary>
        /// Allows the game component to perform any initialization it needs to before starting
        /// to run.  This is where it can query for any required services and load content.
        /// </summary>
        public override void Initialize()
        {
            models = new List<BasicModel>();

            base.Initialize();
        }

        protected override void LoadContent()
        {
            plane = Game.Content.Load<Model>(@"Models\SquarePlane");
            crate = Game.Content.Load<Model>(@"Models\dynamiteCrate01n");

            texture = Game.Content.Load<Texture2D>(@"Textures\dynamiteCrate_diffuse");

            BasicModel bm = new BasicModel(crate, texture, 1.0f, 0);

            foreach(ModelMesh mm in crate.Meshes)
            {
                foreach(ModelMeshPart mp in mm.MeshParts)
                {
                    VertexPositionNormalTexture[] data = new VertexPositionNormalTexture[mp.VertexBuffer.VertexCount];
                    Matrix rotation = Matrix.CreateRotationX(-MathHelper.PiOver2);
                    mp.VertexBuffer.GetData<VertexPositionNormalTexture>(data);
                    for (int i = 0; i < mp.VertexBuffer.VertexCount; i++ )
                    {
                        //data[i].Position = Vector3.Transform(data[i].Position, rotation);
                        data[i].Normal = Vector3.Transform(data[i].Normal, rotation);
                    }
                    mp.VertexBuffer.SetData<VertexPositionNormalTexture>(data);
                }
            }
            
            models.Add(bm);
            models.Add(new BasicModel(Game.Content.Load<Model>(@"Models\dynamiteCrate01n"), texture, 1.0f, 0));
            models[1].Translate(new Vector3(60.0f, 0.0f, 0.0f));
            base.LoadContent();
        }

        /// <summary>
        /// Allows the game component to update itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        public override void Update(GameTime gameTime)
        {
            ProcessKeys();

            UpdateModels(gameTime);

            base.Update(gameTime);
        }

        public override void Draw(GameTime gameTime)
        {
            for (int i = 0; i < models.Count; i++)
            {
                models[i].Draw(((ModelTestGame)Game).camera);
            }

            base.Draw(gameTime);
        }

        private void UpdateModels(GameTime gameTime)
        {
            for (int i = 0; i < models.Count; i++)
            {
                models[i].Update(gameTime);
            }
        }

        private void ProcessKeys()
        {
            BasicModel bm = models[0];
            if (bm == null) return;

            if (Keyboard.GetState().IsKeyDown(Keys.W))
            {
                bm.Translate(new Vector3(0.0f, 0.0f, -1.0f)*speed);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.S))
            {
                bm.Translate(new Vector3(0.0f, 0.0f, 1.0f) * speed);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.A))
            {
                bm.Translate(new Vector3(-1.0f, 0.0f, 0.0f) * speed);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.D))
            {
                bm.Translate(new Vector3(1.0f, 0.0f, 0.0f) * speed);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.Q))
            {
                bm.Translate(new Vector3(0.0f, 1.0f, 0.0f) * speed);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.Z))
            {
                bm.Translate(new Vector3(0.0f, -1.0f, 0.0f) * speed);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.E))
            {
                bm.Rotate(new Vector3(0.0f, MathHelper.PiOver4 / 2.0f, 0.0f) * speed);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.R))
            {
                bm.Rotate(new Vector3(MathHelper.PiOver4 / 2.0f, 0.0f, 0.0f) * speed);
            }
            if (Keyboard.GetState().IsKeyDown(Keys.T))
            {
                bm.Rotate(new Vector3(0.0f, 0.0f, MathHelper.PiOver4 / 2.0f) * speed);
            }
        }
    }
}
