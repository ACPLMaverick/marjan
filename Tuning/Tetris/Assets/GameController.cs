using UnityEngine;
using System.Collections;

public class GameController : MonoBehaviour {

    public static float distance = 0.4f;
    public static Vector3 startPos = new Vector3(-0.1989098f, 4.739200f - distance, 0.0f);
    public static Vector3 wpizdu = new Vector3(100f, 100f, 0.0f);
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
                    if (!checkForEdgeCollisions("L")) currentBlock.transform.position = newPos;
                }

                if (Input.GetKeyDown(KeyCode.RightArrow))
                {
                    Vector3 newPos = new Vector3(currentBlock.transform.position.x + distance, currentBlock.transform.position.y,
                        currentBlock.transform.position.z);
                    if (!checkForEdgeCollisions("R")) currentBlock.transform.position = newPos;
                }

                if (Input.GetKeyDown(KeyCode.Space))
                {
                    Vector3 rotationVector = new Vector3(0.0f, 0.0f, -90.0f);
                    if (true) currentBlock.GetComponent<BlockController>().BlockRotate(rotationVector);
                }
            }
            else
            {
                //COLLISION!
                //Debug.Log("kolizja " + Time.time.ToString());

                collision = false;

                brickPile.GetComponent<BrickPileController>().AddToPile(currentBlock.GetComponent<BlockController>().GetMyBricks());
                int newScore = brickPile.GetComponent<BrickPileController>().CheckAndRemove();
                score += 10*newScore; // tu dodajemy punkty

                createBlock();

                UITextScore.guiText.text = "Score: " + score.ToString();
            }
        }
	}

    void createBlock()
    {
        string type;
        int randomCurrent = (int)Random.value % 7;
        int randomNext = (int)Random.value % 7;
        
        currentBlock = (GameObject)Instantiate(GameObject.Find("blockIshape"), startPos, Quaternion.identity);
    }

    bool checkForCollisions()
    {
        BrickController[] brickControllers = currentBlock.GetComponent<BlockController>().GetMyBricks();

        foreach (BrickController brick in brickControllers)
        {
            RaycastHit2D hit = Physics2D.Raycast(new Vector2(brick.transform.position.x, brick.transform.position.y - 0.2f), new Vector2(Vector3.down.x, Vector3.down.y), 100f);
            //Debug.Log(Mathf.Abs(hit.transform.position.y - brick.transform.position.y));

            bool notOurs = true;
            foreach(BrickController other in brickControllers)
            {
                if (other.GetComponent<BoxCollider2D>() == hit.collider)
                {
                    notOurs = false;
                }
            }

            if(brick != null)
            {
                if ((Mathf.Abs(hit.transform.position.y - brick.transform.position.y) <= (distance + 0.01f)) && notOurs)
                {
                    return true;
                }
            }
        }

        return false;
    }

    bool checkForEdgeCollisions(string side)
    {
        Vector3 oldpos = currentBlock.transform.position;
        //currentBlock.transform.position = pos;
        BrickController[] brickControllers = currentBlock.GetComponent<BlockController>().GetMyBricks();

        Vector2 vector;
        Vector2 sourceVector;

        foreach(BrickController brick in brickControllers)
        {
            if (side == "L")
            {
                vector = new Vector2(Vector3.left.x, Vector3.left.y);
                sourceVector = new Vector2(brick.transform.position.x - (distance / 2 + 0.01f), brick.transform.position.y);
            }
            else if (side == "R")
            {
                vector = new Vector2(Vector3.right.x, Vector3.right.y);
                sourceVector = new Vector2(brick.transform.position.x + (distance / 2 + 0.01f), brick.transform.position.y);
            }
            else return false;
            RaycastHit2D hit = Physics2D.Raycast(sourceVector, vector, 100f);

                Debug.Log((Mathf.Abs(hit.transform.position.x - brick.transform.position.x)));
                if ((Mathf.Abs(hit.transform.position.x - brick.transform.position.x) <= (distance + 0.2f)))
                {
                    //Debug.Log(hit.transform.position.x);
                    bool notOurs = true;

                    foreach(BrickController other in brickControllers)
                    {
                        if (other.GetComponent<BoxCollider2D>() == hit.collider) notOurs = false;
                    }

                    if(notOurs) return true;
                }
        }

        //ArrayList leftBricks = new ArrayList();
        //ArrayList rightBricks = new ArrayList();
        //float currentY = -99.0f;

        //BrickController temp;
        //for (int write = 0; write < brickControllers.Length; write++)
        //{
        //    for (int sort = 0; sort < brickControllers.Length - 1; sort++)
        //    {
        //        if (brickControllers[sort].transform.position.y > brickControllers[sort + 1].transform.position.y)
        //        {
        //            temp = brickControllers[sort + 1];
        //            brickControllers[sort + 1] = brickControllers[sort];
        //            brickControllers[sort] = temp;
        //        }
        //    }
        //}

        //ArrayList tempList = new ArrayList();
        //currentY = brickControllers[0].transform.position.y;
        //foreach(BrickController brick in brickControllers)
        //{
        //    if(brick.transform.position.y != currentY)
        //    {
        //        currentY = brick.transform.position.y;
        //        leftBricks.Add(GetLeastXValue(tempList));
        //        rightBricks.Add(GetMostXValue(tempList));
        //        tempList.Clear();
        //        tempList.Add(brick);
        //    }
        //    else
        //    {
        //        tempList.Add(brick);
        //    }
        //}
        //if(tempList.Count != 0)
        //{
        //    leftBricks.Add(GetLeastXValue(tempList));
        //    rightBricks.Add(GetMostXValue(tempList));
        //}

        //Debug.Log("L: " + leftBricks.Count + " R: " + rightBricks.Count);

        //if(side == "L")
        //{
        //    foreach (Object obj in leftBricks)
        //    {
        //        if (((BrickController)obj).checkEdgeCollision(side)) return true;
        //    }
        //}
        //else if(side == "R")
        //{
        //    foreach (Object obj in rightBricks)
        //    {
        //        if (((BrickController)obj).checkEdgeCollision(side)) return true;
        //    }
        //}

        return false;
    }

    bool checkForEdgeCollisionsRotation(Vector3 pos)
    {
        ArrayList brickVectors = new ArrayList();
        BrickController[] brickControllers = currentBlock.GetComponent<BlockController>().GetMyBricks();

        currentBlock.transform.rotation = Quaternion.Euler(currentBlock.transform.rotation.eulerAngles + pos);

        foreach(BrickController brick in brickControllers)
        {
            brickVectors.Add(brick.transform.position);
        }

        currentBlock.transform.rotation = Quaternion.Euler(currentBlock.transform.rotation.eulerAngles - pos);
        Vector3 oldpos = currentBlock.transform.position;
        currentBlock.transform.position = wpizdu;

        foreach(object obj in brickVectors)
        {
            Vector3 myVector = (Vector3)obj;
            RaycastHit2D hit = Physics2D.Raycast(new Vector2(myVector.x, myVector.y), new Vector2(Vector3.down.x, Vector3.down.y), 100.0f);
            //Debug.Log("VectorX: " + myVector.x + " VectorY: " + myVector.y + " distance: " + hit.distance + " fraction: " + hit.fraction);

            if (hit.fraction <= 0.1)
            {
                currentBlock.transform.position = oldpos;
                return true;
            }
        }

        currentBlock.transform.position = oldpos;
        return false;
    }

    BrickController GetMostXValue(ArrayList list)
    {
        BrickController tempBrick = (BrickController)(list[0]);
        foreach(Object obj in list)
        {
            if (((BrickController)obj).transform.position.x > tempBrick.transform.position.x) tempBrick = (BrickController)obj;
        }
        return tempBrick;
    }

    BrickController GetLeastXValue(ArrayList list)
    {
        BrickController tempBrick = (BrickController)(list[0]);
        foreach (Object obj in list)
        {
            if (((BrickController)obj).transform.position.x < tempBrick.transform.position.x) tempBrick = (BrickController)obj;
        }
        return tempBrick;
    }
}
