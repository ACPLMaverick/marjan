using System;
using System.Diagnostics;
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
    /// This is a game component that implements IUpdateable.
    /// </summary>
    public class Camera : Microsoft.Xna.Framework.GameComponent
    {
        public Matrix ViewMatrix
        {
            get;
            protected set;
        }

        public Matrix ProjectionMatrix
        {
            get;
            protected set;
        }

        public Vector3 Position { get; set; }
        public Vector3 Direction { get; protected set; }
        public Vector3 Up { get; protected set; }
        public Vector3 Right { get; protected set; }
        public float Speed { get; protected set; }

        private float totalYaw = MathHelper.PiOver4 - 0.01f;
        private float currentYaw = 0.0f;
        private float tempYaw;
        private float totalPitch = MathHelper.PiOver2 - 1.0f;
        private float currentPitch = 0.0f;
        private float tempPitch;

        public Camera(Game game, Vector3 pos, Vector3 target, Vector3 up, float speed)
            : base(game)
        {
            Position = pos;
            Direction = target - pos;
            Direction.Normalize();
            Up = up;
            Right = Vector3.Cross(Direction, Up);
            Speed = speed;

            CreateLookAt();

            ProjectionMatrix = Matrix.CreatePerspectiveFieldOfView
            (
                MathHelper.PiOver4,
                (float)Game.Window.ClientBounds.Width/(float)Game.Window.ClientBounds.Height,
                1.0f,
                3000.0f
            );
        }

        /// <summary>
        /// Allows the game component to perform any initialization it needs to before starting
        /// to run.  This is where it can query for any required services and load content.
        /// </summary>
        public override void Initialize()
        {
            // TODO: Add your initialization code here

            base.Initialize();
        }

        /// <summary>
        /// Allows the game component to update itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        public override void Update(GameTime gameTime)
        {
            tempYaw = -MathHelper.PiOver4 / 45.0f * (GamePad.GetState(PlayerIndex.One).ThumbSticks.Left.X);
            tempPitch = MathHelper.PiOver4 / 135.0f * (GamePad.GetState(PlayerIndex.One).ThumbSticks.Left.Y);
            Right = Vector3.Cross(Direction, Up);

            if (Math.Abs(currentPitch + tempPitch) < totalPitch)
            {
                currentPitch += tempPitch;
                Direction = Vector3.Transform(Direction,
                    Matrix.CreateFromAxisAngle(Right, tempPitch));
            }

                currentYaw += tempYaw;
                Direction = Vector3.Transform(Direction,
                    Matrix.CreateFromAxisAngle(Up, tempYaw));

            Direction.Normalize();
          
            CreateLookAt();

            base.Update(gameTime);
        }

        public Vector3 GetDirection() { return Direction; }

        private void CreateLookAt()
        {
            ViewMatrix = Matrix.CreateLookAt(Position, Position + Direction, Up);
        }
    }
}
