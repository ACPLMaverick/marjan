using UnityEngine;
using System.Collections;

public class FluidContainer : MonoBehaviour {

	public double elasticity;
	public float containerBase;
	public float containerHeight;

	private BoxCollider2D myCollider;

	// Use this for initialization
	void Start () {
		myCollider = GetComponent<BoxCollider2D> ();
	}
	
	// Update is called once per frame
	void Update () {
		myCollider.size = new Vector2 (containerBase/10, containerHeight/10);
	}

	public void OnCollisionEnter2D(Collision2D other)
	{
		 
	}
}
