using UnityEngine;
using System.Collections;

public class StarGather : MonoBehaviour {

    public PlayerController player;

	// Use this for initialization
	void Start () {

	}
	
	// Update is called once per frame
	void Update () {
	
	}

    void OnTriggerEnter2D(Collider2D col)
    {
        if (col.gameObject.tag == "Player")
        {
            player.AddScore(1);
            Destroy(this.gameObject);
        }
    }
}
