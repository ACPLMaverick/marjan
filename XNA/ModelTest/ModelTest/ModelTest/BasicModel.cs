using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

namespace ModelTest
{
    public class BasicModel
    {
        public Model MyModel { get; protected set; }
        public Texture2D MyTexture { get; protected set; }
        protected Matrix WorldMatrix;
        protected Matrix rotationMatrix;
        protected Matrix preRotationMatrix;
        protected float scale;
        public int ID;

        public BasicModel(Model m, Texture2D tex, float scale, int ID)
        {
            MyModel = m;
            MyTexture = tex;
            WorldMatrix = Matrix.Identity;
            rotationMatrix = Matrix.Identity;
            preRotationMatrix = Matrix.CreateRotationX(MathHelper.PiOver2);
            this.scale = scale;
            this.ID = ID;

            foreach(ModelMesh mm in MyModel.Meshes)
            {
                foreach(ModelMeshPart mp in mm.MeshParts)
                {
                    foreach (BasicEffect be in mm.Effects)
                    {
                        be.EnableDefaultLighting();
                        be.TextureEnabled = true;
                        be.Texture = MyTexture;
                    }
                }
            }
        }

        public virtual void Update(GameTime time)
        {

        }

        public void Draw(Camera camera)
        {
            Matrix[] transforms = new Matrix[MyModel.Bones.Count];
            MyModel.CopyAbsoluteBoneTransformsTo(transforms);

            foreach(ModelMesh mm in MyModel.Meshes)
            {
                foreach(BasicEffect be in mm.Effects)
                {
                    be.Projection = camera.ProjectionMatrix;
                    be.View = camera.ViewMatrix;
                    be.World = Matrix.CreateScale(scale) * GetWorldMatrix();
                    be.Texture = MyTexture;
                }

                mm.Draw();
            }
        }

        public void Translate(Vector3 vec)
        {
            WorldMatrix *= Matrix.CreateTranslation(vec);
        }

        public void Rotate(Vector3 vec)
        {
            rotationMatrix *= Matrix.CreateFromYawPitchRoll(vec.X, vec.Y, vec.Z);
        }

        public virtual Matrix GetWorldMatrix() { return rotationMatrix*WorldMatrix/**preRotationMatrix*/; }
    }
}
