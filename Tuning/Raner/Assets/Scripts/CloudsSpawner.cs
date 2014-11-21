using UnityEngine;
using System.Collections;

public class CloudsSpawner : MonoBehaviour {

    public GameObject cloud;
    private GameObject player;
    private GameObject clone;
    private Vector3 spawnPoint;
    private float spawnTime = 4.0f;
    private float isSpawned;
    private bool isAlive;
    private float speed = 0.05f;


    // Use this for initialization
    void Start()
    {
        clone = GameObject.FindWithTag("Cloud");
        player = GameObject.FindWithTag("Player");
        InvokeRepeating("Spawn", spawnTime, spawnTime);
    }

    void Update()
    {
        if (clone == null) clone = GameObject.FindWithTag("Cloud");
        clone.transform.Translate(-this.transform.right * speed);
    }

    void Spawn()
    {
        isSpawned = Random.Range(0.0f, 1.0f);
        spawnPoint = new Vector3(player.transform.position.x + 20.0f, 2.5f, -1.0f);
        if (isSpawned > 0.15f)
        {
            clone = (GameObject)Instantiate(cloud, spawnPoint, Quaternion.identity);
            Destroy(clone, 8.0f);
        }
    }
}
