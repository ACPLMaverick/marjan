using UnityEngine;
using System.Collections;

public class EnemySpawn : MonoBehaviour {

    public GameObject enemy = null;
    public GameObject obstacle = null;
    //private Vector3 spawnPoint;
    private float spawnTime = 3.0f;
    private bool isAlive;
    private float speed = 0.15f;
    private GameObject player;
    private GameObject clone;
    private GameObject obstacleClone;

	// Use this for initialization
	void Start () {
        clone = GameObject.FindWithTag("Enemy");
        obstacleClone = GameObject.FindWithTag("Obstacle");
        player = GameObject.FindWithTag("Player");
        if(enemy != null) InvokeRepeating("Spawn", spawnTime, spawnTime);
        if(obstacle != null) InvokeRepeating("ObstacleSpawn", spawnTime, spawnTime);
	}

    void Update()
    {
        if (clone == null) clone = GameObject.FindWithTag("Enemy");
        clone.transform.Translate(-this.transform.right * speed);
    }
    
    void Spawn()
    {
        float isSpawned = Random.Range(0.0f, 1.0f);
        Vector3 spawnPoint = new Vector3(player.transform.position.x + 20.0f, player.transform.position.y, -1.5f);
        if (isSpawned > 0.5f)
        {
            clone = (GameObject)Instantiate(enemy, spawnPoint, Quaternion.identity);
            Destroy(clone, 8.0f);
        }
    }

    void ObstacleSpawn()
    {
        float isSpawned = Random.Range(0.0f, 1.0f);
        Vector3 spawnPoint;
        Quaternion spawnRotate;
        if (isSpawned > 0.5f && isSpawned <= 0.75f)
        {
            spawnPoint = new Vector3(player.transform.position.x + 20.0f, -3.5f, -1.5f);
            obstacleClone = (GameObject)Instantiate(obstacle, spawnPoint, Quaternion.identity);
            Destroy(obstacleClone, 8.0f);
        }
        if(isSpawned > 0.75f)
        {
            spawnPoint = new Vector3(player.transform.position.x + 20.0f, 3.0f, -1.5f);
            spawnRotate = Quaternion.Euler(0, 0, 180);
            obstacleClone = (GameObject)Instantiate(obstacle, spawnPoint, spawnRotate);
            Destroy(obstacleClone, 8.0f);
        }
    }
}
