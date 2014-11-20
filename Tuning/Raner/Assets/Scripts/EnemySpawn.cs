using UnityEngine;
using System.Collections;

public class EnemySpawn : MonoBehaviour {

    public GameObject enemy;
    private Vector3 spawnPoint;
    private float spawnTime = 3.0f;
    private float isSpawned;
    private bool isAlive;
    private float speed = 0.15f;
    private GameObject player;
    private GameObject clone;

	// Use this for initialization
	void Start () {
        clone = GameObject.FindWithTag("Enemy");
        player = GameObject.FindWithTag("Player");
        InvokeRepeating("Spawn", spawnTime, spawnTime);
	}

    void Update()
    {
        if (clone == null) clone = GameObject.FindWithTag("Enemy");
        clone.transform.Translate(-this.transform.right * speed);
    }
    
    void Spawn()
    {
        isSpawned = Random.Range(0.0f, 1.0f);
        Debug.Log(isSpawned);
        spawnPoint = new Vector3(player.transform.position.x + 20.0f, player.transform.position.y, -1.5f);
        if (isSpawned > 0.5f)
        {
            clone = (GameObject)Instantiate(enemy, spawnPoint, Quaternion.identity);
            Destroy(clone, 8.0f);
        }
    }
}
