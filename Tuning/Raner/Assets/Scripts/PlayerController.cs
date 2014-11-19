﻿using UnityEngine;
using System.Collections;

public class PlayerController : MonoBehaviour {

	private float speed;

	private float moveHorizontal;
	private float moveVertical;
	private Vector2 movement;
	private Vector2 dashVector;

	public GameObject star;
	public GUIText scoreText;
	private int score;

	// Use this for initialization
	void Start () {
		speed = 25.0f;
		dashVector = new Vector2(25000.0f, 0.0f);
		star = GetComponent<PlayerController> ().star;
		score = 0;
		UpdateScore ();
	}
	
	// Update is called once per frame
	void Update () {
		moveHorizontal = Input.GetAxis ("Horizontal");
		moveVertical = Input.GetAxis ("Vertical");
		movement = new Vector2 (moveHorizontal, moveVertical);
		this.rigidbody2D.velocity = movement * speed;

		this.rigidbody2D.position = new Vector2(Mathf.Clamp(rigidbody2D.position.x, Camera.main.gameObject.transform.position.x - 12.0f, Camera.main.gameObject.transform.position.x + 8.0f), Mathf.Clamp (rigidbody2D.position.y, -3.5f, 3.5f));
		if(Input.GetKeyUp(KeyCode.Space))
		{
			this.rigidbody2D.AddForceAtPosition(dashVector, rigidbody2D.position);
		}
	}

	void OnCollisionEnter2D(Collision2D col)
	{
		if(col.gameObject.tag == "Star")
		{
			Destroy(star);
			AddScore(1);
		}
	}

	public void AddScore(int newScoreValue)
	{
		score += newScoreValue;
		UpdateScore ();
	}
	
	void UpdateScore()
	{
		scoreText.text = "Score: " + score.ToString();
	}
}