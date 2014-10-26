using UnityEngine;
using System.Collections;

public class UpperColliderController : MonoBehaviour {

    GameObject brickPile = null;

	// Use this for initialization
	void Start () {
        brickPile = GameObject.Find("BrickPile");
	}
	
	// Update is called once per frame
	void Update () {
	
	}

    void OnCollisionEnter2D(Collision2D other)
    {
        Debug.Log("end");
        Application.Quit();
    }
}
