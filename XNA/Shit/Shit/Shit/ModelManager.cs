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


namespace Shit
{
    /// <summary>
    /// This is a game component that implements IUpdateable.
    /// </summary>
    public class ModelManager : IGameComponent, IDrawable, IUpdateable
    {
        private List<BasicModel> models;
        private List<BasicModel> bullets;
        private Model plane;
        private Model crate;

        Random rnd;

        Vector3 maxSpawnLocation = new Vector3(100.0f, 100.0f, -3000.0f);
        int nextSpawnTime = 0;
        int timeSinceLastSpawn = 0;
        float maxRollAngle = MathHelper.Pi / 40.0f;

        int enemiesThisLevel = 0;
        int missedThisLevel = 0;
        int currentLevel = 0;

        float shotMinZ = -3000.0f;

        List<LevelInfo> levelInfoList = new List<LevelInfo>();

        public ModelManager(Game game)
            : base(game)
        {
            rnd = ((ShitGame)Game).random;
        }

        /// <summary>
        /// Allows the game component to perform any initialization it needs to before starting
        /// to run.  This is where it can query for any required services and load content.
        /// </summary>
        public override void Initialize()
        {
            models = new List<BasicModel>();
            bullets = new List<BasicModel>();

            //initialize levels
            levelInfoList.Add(new LevelInfo(1000, 3000, 20, 2, 6, 10));
            levelInfoList.Add(new LevelInfo(900, 2800, 22, 2, 6, 9));
            levelInfoList.Add(new LevelInfo(800, 2600, 24, 2, 6, 8));
            levelInfoList.Add(new LevelInfo(700, 2400, 26, 3, 7, 7));
            levelInfoList.Add(new LevelInfo(600, 2200, 28, 3, 7, 6));
            levelInfoList.Add(new LevelInfo(500, 2000, 30, 3, 7, 5));
            levelInfoList.Add(new LevelInfo(400, 1800, 32, 4, 7, 4));
            levelInfoList.Add(new LevelInfo(300, 1600, 34, 4, 8, 3));
            levelInfoList.Add(new LevelInfo(200, 1400, 36, 5, 8, 2));
            levelInfoList.Add(new LevelInfo(100, 1200, 38, 5, 9, 1));
            levelInfoList.Add(new LevelInfo(50, 1000, 40, 6, 9, 0));
            levelInfoList.Add(new LevelInfo(50, 800, 42, 6, 9, 0));
            levelInfoList.Add(new LevelInfo(50, 600, 44, 8, 10, 0));
            levelInfoList.Add(new LevelInfo(25, 400, 46, 8, 10, 0));
            levelInfoList.Add(new LevelInfo(0, 200, 48, 18, 20, 0));

            SetNextSpawnTime();

            base.Initialize();
        }

        protected override void LoadContent()
        {
            plane = Game.Content.Load<Model>(@"Models\SquarePlane");
            crate = Game.Content.Load<Model>(@"Models\TypowaSfera");
            base.LoadContent();
        }

        /// <summary>
        /// Allows the game component to update itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        public override void Update(GameTime gameTime)
        {
            CheckToSpawnEnemy(gameTime);

            UpdateModels(gameTime);

            UpdateBullets(gameTime);

            if(enemiesThisLevel == levelInfoList[currentLevel].NumberEnemies && models.Count == 0)
            {
                currentLevel++;
                ((ShitGame)Game).level++;
            }

            base.Update(gameTime);
        }

        public override void Draw(GameTime gameTime)
        {
            for (int i = 0; i < models.Count; i++)
            {
                models[i].Draw(((ShitGame)Game).camera);
            }
            for (int i = 0; i < bullets.Count; i++)
            {
                bullets[i].Draw(((ShitGame)Game).camera);
            }

            base.Draw(gameTime);
        }

        public void AddBullet(Vector3 position, Vector3 direction)
        {
            bullets.Add(new SpinningEnemy(crate, ((ShitGame)Game).texture_c, position, direction, 0, 0, 0.1f, 0.3f, -1));
            Debug.WriteLine(bullets.Count.ToString());
        }

        private void UpdateModels(GameTime gameTime)
        {
            for (int i = 0; i < models.Count; i++)
            {
                models[i].Update(gameTime);

                if(models[i].GetWorldMatrix().Translation.Z > ((ShitGame)Game).camera.Position.Z + 100)
                {
                    models.RemoveAt(i);
                    --i;
                }
            }
        }

        private void UpdateBullets(GameTime gameTime)
        {
            for (int i = 0; i < bullets.Count; i++)
            {
                bullets[i].Update(gameTime);

                if (bullets[i].GetWorldMatrix().Translation.Z < shotMinZ)
                {
                    bullets.RemoveAt(i);
                    --i;
                }
                else
                {
                    for (int j = 0; j < models.Count; j++)
                    {
                        if (bullets[i].CollidesWith(models[j].MyModel, models[j].GetWorldMatrix()))
                        {
                            // collision!
                            ((ShitGame)Game).PlayCue("death");
                            if (models[j].ID == 0) ((ShitGame)Game).points_p++;
                            else if (models[j].ID == 1) ((ShitGame)Game).points_m++;

                            models.RemoveAt(j);
                            bullets.RemoveAt(i);
                            --i;
                            break;
                        }
                    }
                }
            }  
        }

        private void SetNextSpawnTime()
        {
            nextSpawnTime = rnd.Next(
                levelInfoList[currentLevel].MinSpawnTime,
                levelInfoList[currentLevel].MaxSpawnTime
                );
            timeSinceLastSpawn = 0;
        }

        private void SpawnEnemy()
        {
            Vector3 position = new Vector3(rnd.Next(-(int)maxSpawnLocation.X, (int)maxSpawnLocation.X), rnd.Next(-(int)maxSpawnLocation.Y, (int)maxSpawnLocation.Y), (int)maxSpawnLocation.Z);
            Vector3 direction = new Vector3(0, 0, rnd.Next(levelInfoList[currentLevel].MinSpeed, levelInfoList[currentLevel].MaxSpeed));
            float rollRotation = (float)(rnd.NextDouble() * maxRollAngle - (maxRollAngle / 2));
            int randTex = rnd.Next(2);
            if (randTex == 0) models.Add(new SpinningEnemy(plane, ((ShitGame)Game).texture, position, direction, 0.5f, 0.5f, rollRotation, 2.0f, 0));
            else models.Add(new SpinningEnemy(plane, ((ShitGame)Game).texture_m, position, direction, 0.5f, 0.5f, rollRotation, 1.5f, 1));

            ++enemiesThisLevel;
            SetNextSpawnTime();
        }

        private void CheckToSpawnEnemy(GameTime gameTime)
        {
            if(enemiesThisLevel < levelInfoList[currentLevel].NumberEnemies)
            {
                timeSinceLastSpawn += gameTime.ElapsedGameTime.Milliseconds;
                if(timeSinceLastSpawn > nextSpawnTime)
                {
                    SpawnEnemy();
                }
            }
        }
    }
}
