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
    public class GameObject : Microsoft.Xna.Framework.GameComponent
    {
        public readonly string name;
        public Texture2D texture;
        public Vector2 position;
        public float rotation;
        public float scale;
        public Color color;

        public Vector2 speed;

        public GameObject(Game game, string name, Texture2D texture, Vector2 position, float rotation, float scale, Color color)
            : base(game)
        {
            this.name = name;
            this.texture = texture;
            this.position = position;
            this.rotation = rotation;
            this.scale = scale;
            this.color = color;
            this.speed = Vector2.Zero;
        }

        /// <summary>
        /// Allows the game component to perform any initialization it needs to before starting
        /// to run.  This is where it can query for any required services and load content.
        /// </summary>
        public override void Initialize()
        {

            base.Initialize();
        }

        /// <summary>
        /// Allows the game component to update itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        public override void Update(GameTime gameTime)
        {
            if(speed != Vector2.Zero)
            {
                int time = gameTime.ElapsedGameTime.Milliseconds;
                this.position = this.position + speed * (float)time;
            }

            base.Update(gameTime);
        }
    }
}
