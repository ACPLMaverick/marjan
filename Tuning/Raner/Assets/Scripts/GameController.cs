using UnityEngine;
using System.Collections;

public class GameController : MonoBehaviour {

	private float speed;
	private float x;
	private ArrayList backgrounds = new ArrayList();

	public GameObject bg1, bg2, bg3;

	// Use this for initialization
	void Start () {
		speed = 0.1f;
		x = 4.0f;
		backgrounds.Add (bg1);
		backgrounds.Add (bg2);
		backgrounds.Add (bg3);
	}
	
	// Update is called once per frame
	void Update () {
		this.transform.Translate(this.transform.right * speed);
		if(Time.time > x)
		{
			CreateBackground ();
		}
	}

	void CreateBackground()
	{
		GameObject thisItem0 = (GameObject)backgrounds[0];
		thisItem0.transform.Translate(57.422693f, 0.0f, 0.0f);
		x += 3.25f;
		backgrounds.RemoveAt (0);
		backgrounds.Insert (2, thisItem0);
	}

}

//22.95219f, 0.007607499f
//19.196271f
// if(currentTime - lastTime > x) ?

