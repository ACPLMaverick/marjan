using UnityEngine;
using System.Collections;

public class BrickController : MonoBehaviour {

    public bool collision;
    public bool edgeCollision;

    public static float edgeLeft;
    public static float edgeRight;
    public static float edgeBottom;

	// Use this for initialization
	void Start () {
        collision = false;
        edgeLeft = GameObject.Find("EdgeColliderLeft").transform.position.x;
        edgeRight = GameObject.Find("EdgeColliderRight").transform.position.x;
        edgeBottom = GameObject.Find("BottomCollider").transform.position.y;
	}
	
	// Update is called once per frame
	void Update () {

	}

    public bool checkEdgeCollision(string side)
    {
        //if (transform.position.x < edgeLeft || transform.position.x > edgeRight) edgeCollision = true;
        //else edgeCollision = false;
        edgeCollision = false;
        Vector2 vector;
        Vector2 sourceVector;
        if (side == "L")
        {
            vector = new Vector2(Vector3.left.x, Vector3.left.y);
            sourceVector = new Vector2(transform.position.x - (GameController.distance / 2 + 0.01f), transform.position.y);
        }
        else if (side == "R")
        {
            vector = new Vector2(Vector3.right.x, Vector3.right.y);
            sourceVector = new Vector2(transform.position.x + (GameController.distance / 2 + 0.01f), transform.position.y);
        }
        else return edgeCollision;

        RaycastHit2D hit = Physics2D.Raycast(sourceVector, vector, 100f);
 
        if(hit != null && gameObject != null)
        {
            if ((Mathf.Abs(hit.transform.position.x - transform.position.x) <= (GameController.distance + 0.01f)))
            {
                //Debug.Log(hit.transform.position.x);
                edgeCollision = true;
            }
        }
        

        return edgeCollision;
    }

    public bool checkCollision()
    {
        //if (transform.position.y < edgeBottom) collision = true;
        return collision;
    }

    // niepotrzebne

    void OnCollisionEnter2D(Collision2D col)
    {
        if ((col.gameObject.name == "SingleBrick") || (col.gameObject.name == "BottomCollider"))
        {
            collision = true;
            Debug.Log("Collision with brick/bottom!");
        }
        else if (col.gameObject.name == "EdgeCollider")
        {
            edgeCollision = true;
            Debug.Log("Collision with edge!");
        }
        Debug.Log("Collision with something else!");
    }

    void OnCollisionExit2D(Collision2D col)
    {
        if ((col.gameObject.name == "SingleBrick") || (col.gameObject.name == "BottomCollider"))
        {
            collision = false;
            Debug.Log("Exiting Collision with brick/bottom!");
        }
        else if (col.gameObject.name == "EdgeCollider")
        {
            edgeCollision = false;
            Debug.Log("Exiting Collision with edge!");
        }
        Debug.Log("Exiting Collision with something else!");
    }

    void OnCollisionEnter(Collision col)
    {
        if ((col.gameObject.name == "SingleBrick") || (col.gameObject.name == "BottomCollider"))
        {
            collision = true;
            Debug.Log("Collision with brick/bottom!");
        }
        else if (col.gameObject.name == "EdgeCollider")
        {
            edgeCollision = true;
            Debug.Log("Collision with edge!");
        }
        Debug.Log("Collision with something else!");
    }

    void OnCollisionExit(Collision col)
    {
        if ((col.gameObject.name == "SingleBrick") || (col.gameObject.name == "BottomCollider"))
        {
            collision = false;
            Debug.Log("Exiting Collision with brick/bottom!");
        }
        else if (col.gameObject.name == "EdgeCollider")
        {
            edgeCollision = false;
            Debug.Log("Exiting Collision with edge!");
        }
        Debug.Log("Exiting Collision with something else!");
    }
}
