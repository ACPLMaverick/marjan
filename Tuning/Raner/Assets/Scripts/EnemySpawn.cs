using UnityEngine;
using System.Collections;

public class EnemySpawn : MonoBehaviour {

    public GameObject enemy;
    private Vector3 spawnPoint;
    private float spawnTime = 3.0f;
    private float isSpawned;
    private GameObject player;

	// Use this for initialization
	void Start () {
        player = GameObject.FindWithTag("Player");
        InvokeRepeating("Spawn", spawnTime, spawnTime);
	}

    void Spawn()
    {
        isSpawned = Random.Range(0.0f, 1.0f);
        Debug.Log(isSpawned);
        spawnPoint = new Vector3(player.transform.position.x + 20.0f, player.transform.position.y, -1.5f);
        if(isSpawned > 0.5f) Instantiate(enemy, spawnPoint, Quaternion.identity);
    }
}
