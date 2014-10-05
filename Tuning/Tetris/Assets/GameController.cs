using UnityEngine;
using System.Collections;

public class GameController : MonoBehaviour {

    public static float distance = 0.4f;
    public static Vector3 startPos = new Vector3(-0.1989098f, 4.739200f - distance, 0.0f);
    public static float falldownSpeed = 4.0f;
    float refTime;
    bool collision;
    int score = 0;

    GameObject currentBlock = null;
    GameObject brickPile = null;
    GameObject UITextNextblock;
    GameObject UITextScore;

	// Use this for initialization
	void Start () {
        createBlock();
        brickPile = GameObject.Find("BrickPile");
        refTime = Time.time;
        collision = false;

        UITextNextblock = GameObject.Find("UITextNextblock");
        UITextScore = GameObject.Find("UITextScore");
        UITextScore.guiText.text = "Score: " + score.ToString();
	}
	
	// Update is called once per frame
	void Update () {
        if (currentBlock != null)
        {
            if(!collision)
            {
                if (Time.time - refTime > (1.0f/falldownSpeed))
                {
                    refTime = Time.time;
                    Vector3 newPos = new Vector3(currentBlock.transform.position.x, currentBlock.transform.position.y - distance,
                        currentBlock.transform.position.z);
                    currentBlock.transform.position = newPos;
                    collision = checkForCollisions();
                }

                if (Input.GetKeyDown(KeyCode.LeftArrow))
                {
                    Vector3 newPos = new Vector3(currentBlock.transform.position.x - distance, currentBlock.transform.position.y,
                        currentBlock.transform.position.z);
                    if (!checkForEdgeCollisions(newPos)) currentBlock.transform.position = newPos;
                }

                if (Input.GetKeyDown(KeyCode.RightArrow))
                {
                    Vector3 newPos = new Vector3(currentBlock.transform.position.x + distance, currentBlock.transform.position.y,
                        currentBlock.transform.position.z);
                    if (!checkForEdgeCollisions(newPos)) currentBlock.transform.position = newPos;
                }

                if (Input.GetKeyDown(KeyCode.Space))
                {
                    Quaternion newRotation = currentBlock.transform.rotation;
                    newRotation = Quaternion.Euler(newRotation.eulerAngles + new Vector3(0.0f, 0.0f, -90.0f));
                    if (!checkForEdgeCollisionsRotation(newRotation)) currentBlock.transform.rotation = newRotation;
                }
            }
            else
            {
                //COLLISION!
                //Debug.Log("kolizja " + Time.time.ToString());

                collision = false;

                brickPile.GetComponent<BrickPileController>().AddToPile(currentBlock.GetComponent<BlockController>().GetMyBricks());
                score += 10*(brickPile.GetComponent<BrickPileController>().CheckAndRemove()); // tu dodajemy punkty

                createBlock();

                UITextScore.guiText.text = "Score: " + score.ToString();
            }
        }
	}

    void createBlock()
    {
        currentBlock = (GameObject)Instantiate(GameObject.Find("blockIshape"), startPos, Quaternion.identity);
    }

    bool checkForCollisions()
    {
        BrickController[] brickControllers = currentBlock.GetComponent<BlockController>().GetMyBricks();

        foreach (BrickController brick in brickControllers)
        {
            if (brick.checkCollision())
            {
                return true;
            }

            RaycastHit2D hit = Physics2D.Raycast(new Vector2(brick.transform.position.x, brick.transform.position.y - (distance / 2)), new Vector2(Vector3.down.x, Vector3.down.y), 100f);
            //Debug.Log(Mathf.Abs(hit.transform.position.y - brick.transform.position.y));

            bool notOurs = true;
            foreach(BrickController other in brickControllers)
            {
                if(other.transform.position.x == hit.transform.position.x)
                {
                    if (other.transform.position.y != brick.transform.position.y) notOurs = false;
                }
            }

            if ((Mathf.Abs(hit.transform.position.y - brick.transform.position.y) <= (distance + 0.01f)) && notOurs)
            {
                return true;
            }
        }

        return false;
    }

    bool checkForEdgeCollisions(Vector3 pos)
    {
        Vector3 oldpos = currentBlock.transform.position;
        currentBlock.transform.position = pos;
        BrickController[] brickControllers = currentBlock.GetComponent<BlockController>().GetMyBricks();

        //GameObject[] pileBricks = brickPile.GetComponent<BrickPileController>().GetMyBricks();
        foreach(BrickController brick in brickControllers)
        {
            if (brick.checkEdgeCollision())
            {
                currentBlock.transform.position = oldpos;
                return true;
            }
            /*
            foreach(GameObject obj in pileBricks)
            {
                Debug.Log(obj.transform.position + " | " + brick.gameObject.transform.position);
                if(Mathf.Abs(obj.transform.position.x - brick.gameObject.transform.position.x) <= 0.1 &&
                    Mathf.Abs(obj.transform.position.y - brick.gameObject.transform.position.y) <= 0.1)
                {
                    currentBlock.transform.position = oldpos;
                    return true;
                }
            }
             */
        }

        currentBlock.transform.position = oldpos;
        return false;
    }

    bool checkForEdgeCollisionsRotation(Quaternion pos)
    {
        Quaternion oldpos = currentBlock.transform.rotation;
        currentBlock.transform.rotation = pos;
        BrickController[] brickControllers = currentBlock.GetComponent<BlockController>().GetMyBricks();

        foreach (BrickController brick in brickControllers)
        {
            if (brick.checkEdgeCollision())
            {
                currentBlock.transform.rotation = oldpos;
                return true;
            }

            //GameObject[] pileBricks = brickPile.GetComponent<BrickPileController>().GetMyBricks();
            //foreach (GameObject obj in pileBricks)
            //{
            //    Debug.Log(obj.transform.position + " | " + brick.gameObject.transform.position);
            //    if (Mathf.Abs(obj.transform.position.x - brick.gameObject.transform.position.x) <= 0.1 &&
            //        Mathf.Abs(obj.transform.position.y - brick.gameObject.transform.position.y) <= 0.1)
            //    {
            //        currentBlock.transform.rotation = oldpos;
            //        return true;
            //    }
            //}
        }

        currentBlock.transform.rotation = oldpos;
        return false;
    }
}
