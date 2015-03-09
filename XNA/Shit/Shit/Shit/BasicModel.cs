using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

namespace Shit
{
    public class BasicModel
    {
        public Model MyModel { get; protected set; }
        public Texture2D MyTexture { get; protected set; }
        protected Matrix WorldMatrix;
        protected float scale;

        public BasicModel(Model m, Texture2D tex, float scale)
        {
            MyModel = m;
            MyTexture = tex;
            WorldMatrix = Matrix.Identity;
            this.scale = scale;

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
                    be.World = Matrix.CreateScale(scale) * GetWorldMatrix() * mm.ParentBone.Transform;
                }

                mm.Draw();
            }
        }

        public bool CollidesWith(Model otherModel, Matrix otherWorld)
        {
            // Loop through each ModelMesh in both objects and compare
            // all bounding spheres for collisions
            foreach (ModelMesh myModelMeshes in MyModel.Meshes)
            {
                foreach (ModelMesh hisModelMeshes in otherModel.Meshes)
                {
                    if (myModelMeshes.BoundingSphere.Transform(GetWorldMatrix()).Intersects(hisModelMeshes.BoundingSphere.Transform(otherWorld)))
                        return true;
                }
            }
            return false;
        }

        public virtual Matrix GetWorldMatrix() { return WorldMatrix; }
    }
}
