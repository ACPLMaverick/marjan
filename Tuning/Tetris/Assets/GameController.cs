using UnityEngine;
using System.Collections;

public class GameController : MonoBehaviour {

    public static float distance = 0.4f;
    public static Vector3 startPos = new Vector3(-0.1989098f, 4.739200f - distance, 0.0f);
    public static Vector3 previewPos = new Vector3(3.717794f, 4.342112f, 0.0f);
    public static Vector3 wpizdu = new Vector3(100f, 100f, 0.0f);
    public static float falldownSpeed = 4.0f;
    float refTime;
    bool collision;
    int score = 0;

    public static Color[] brickColors;

    GameObject currentBlock = null;
    GameObject nextBlock = null;
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

        brickColors = new Color[7] { Color.blue, Color.yellow, Color.red, Color.magenta, Color.cyan, Color.green, Color.clear }; 
        //colorIshape = Color.blue;
        //colorJshape = Color.yellow;
        //colorLshape = Color.red;
        //colorOshape = Color.magenta;
        //colorTshape = Color.cyan;
        //colorSshape = Color.green;
        //colorZshape = Color.clear;
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
                    if (!checkForEdgeCollisionsRotation(rotationVector)) currentBlock.GetComponent<BlockController>().BlockRotate(rotationVector);
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
        string typeCurrent, typeNext;
        int randomCurrent = Random.Range(0, 7);
        int randomNext = Random.Range(0, 7);
        Debug.Log(randomCurrent + " " + randomNext);

        if(currentBlock == null)
        {
            switch (randomCurrent)
            {
                case 0:
                    typeCurrent = "blockIshape";
                    break;
                case 1:
                    typeCurrent = "blockJshape";
                    break;
                case 2:
                    typeCurrent = "blockLshape";
                    break;
                case 3:
                    typeCurrent = "blockOshape";
                    break;
                case 4:
                    typeCurrent = "blockZshape";
                    break;
                case 5:
                    typeCurrent = "blockSshape";
                    break;
                case 6:
                    typeCurrent = "blockTshape";
                    break;
                default:
                    typeCurrent = "blockIshape";
                    break;
            }
            currentBlock = (GameObject)Instantiate(GameObject.Find(typeCurrent), startPos, Quaternion.identity);
            ChangeColor(currentBlock, Color.red);
            switch (randomNext)
            {
                case 0:
                    typeNext = "blockIshape";
                    break;
                case 1:
                    typeNext = "blockJshape";
                    break;
                case 2:
                    typeNext = "blockLshape";
                    break;
                case 3:
                    typeNext = "blockOshape";
                    break;
                case 4:
                    typeNext = "blockZshape";
                    break;
                case 5:
                    typeNext = "blockSshape";
                    break;
                case 6:
                    typeNext = "blockTshape";
                    break;
                default:
                    typeNext = "blockIshape";
                    break;
            }
            nextBlock = (GameObject)Instantiate(GameObject.Find(typeNext), previewPos, Quaternion.identity);
            //ChangeColor(nextBlock, brickColors[randomNext]);
        }
        else
        {
            currentBlock = nextBlock;
            currentBlock.transform.position = startPos;
            switch (randomNext)
            {
                case 0:
                    typeNext = "blockIshape";
                    break;
                case 1:
                    typeNext = "blockJshape";
                    break;
                case 2:
                    typeNext = "blockLshape";
                    break;
                case 3:
                    typeNext = "blockOshape";
                    break;
                case 4:
                    typeNext = "blockZshape";
                    break;
                case 5:
                    typeNext = "blockSshape";
                    break;
                case 6:
                    typeNext = "blockTshape";
                    break;
                default:
                    typeNext = "blockIshape";
                    break;
            }
            nextBlock = (GameObject)Instantiate(GameObject.Find(typeNext), previewPos, Quaternion.identity);
            //ChangeColor(nextBlock, brickColors[randomNext]);
        }
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
                if ((Mathf.Abs(hit.point.y - brick.transform.position.y) <= (0.22f)) && notOurs)
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
                if ((Mathf.Abs(hit.transform.position.x - brick.transform.position.x) <= (2*distance)))
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

            if (hit.fraction <= 0.01)
            {
                currentBlock.transform.position = oldpos;
                return true;
            }
        }

        currentBlock.transform.position = oldpos;
        return false;
    }

    void ChangeColor(GameObject block, Color color)
    {
        BrickController[] bricks = block.GetComponent<BlockController>().GetMyBricks();
        foreach(BrickController brick in bricks)
        {
            brick.renderer.material.color = color;
        }
    }
}
