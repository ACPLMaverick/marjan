using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

namespace Shit
{
    public class Scene
    {
        private List<GameObject> gameObjects;

        public Scene()
        {
            gameObjects = new List<GameObject>();
        }

        public GameObject Add(GameObject obj)
        {
            gameObjects.Add(obj);
            return obj;
        }

        public void DrawAll(SpriteBatch spriteBatch, GameTime gameTime)
        {
            spriteBatch.Begin();
            foreach(GameObject obj in gameObjects)
            {
                spriteBatch.Draw(obj.texture, obj.position, null, obj.color, obj.rotation, Vector2.Zero, obj.scale, SpriteEffects.None, 0.0f);
            }
            spriteBatch.End();
        }

        public void UpdateAll(GameTime gameTime)
        {
            foreach(GameObject obj in gameObjects)
            {
                obj.Update(gameTime);
            }
        }
    }
}
