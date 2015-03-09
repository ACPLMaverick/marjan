using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Shit
{
    public class LevelInfo
    {
        public int MinSpawnTime { get; set; }
        public int MaxSpawnTime { get; set; }

        public int NumberEnemies { get; set; }
        public int MinSpeed { get; set; }
        public int MaxSpeed { get; set; }

        public int MissesAllowed { get; set; }

        public LevelInfo(int MinSpawnTime, int MaxSpawnTime, int NumberEnemies, int MinSpeed, int MaxSpeed, int MissesAllowed)
        {
            this.MinSpawnTime = MinSpawnTime;
            this.MaxSpawnTime = MaxSpawnTime;
            this.NumberEnemies = NumberEnemies;
            this.MinSpeed = MinSpeed;
            this.MaxSpeed = MaxSpeed;
            this.MissesAllowed = MissesAllowed;
        }
    }
}
