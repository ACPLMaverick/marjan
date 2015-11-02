using UnityEngine;
using UnityEngine.EventSystems;
using System.Collections;

public class InteractiveObject : MonoBehaviour {

	public uint ID;

	public Sprite square;
	public Sprite circle;

	public double mass;
	public double elasticity;
	public double velocity;
	
	private SpriteRenderer mySpriteRenderer;

	// Use this for initialization
	void Start () {
		mySpriteRenderer = GetComponent<SpriteRenderer> ();
	}
	
	// Update is called once per frame
	void Update () {
	
	}

	public void SetSprite(int n)
	{
		switch (n) {
		case 0:
			mySpriteRenderer.sprite = square;
			this.gameObject.AddComponent<BoxCollider2D>();
			break;
		case 1:
			mySpriteRenderer.sprite = circle;
			this.gameObject.AddComponent<CircleCollider2D>();
			break;
		}
	}

	public void OnMouseOver()
	{

	}

	public void OnMouseDown()
	{
		if(FluidController.Instance.canDelete)
			FluidController.Instance.DestroyInteractiveObject (this);
	}

	public void OnCollisionEnter2D(Collision2D other)
	{

	}
}
