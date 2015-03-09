using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

namespace Shit
{
    public class SpinningEnemy : BasicModel
    {
        protected Matrix rotationMatrix;
        float yawAngle = 0.0f;
        float pitchAngle = 0.0f;
        float rollAngle = 0.0f;
        Vector3 direction;

        public SpinningEnemy(Model m, Texture2D tex, Vector3 position, Vector3 direction,
            float yaw, float pitch, float roll, float scale) : base(m, tex, scale)
        {
            WorldMatrix = Matrix.CreateTranslation(position);
            rotationMatrix = Matrix.Identity;
            this.direction = direction;
            this.yawAngle = yaw;
            this.pitchAngle = pitch;
            this.rollAngle = roll;
        }

        public override void Update(GameTime time)
        {
            rotationMatrix *= Matrix.CreateFromYawPitchRoll(yawAngle, pitchAngle, rollAngle);
            WorldMatrix *= Matrix.CreateTranslation(direction);
            base.Update(time);
        }

        public override Matrix GetWorldMatrix()
        {
            return rotationMatrix * base.GetWorldMatrix();
        }
    }
}
