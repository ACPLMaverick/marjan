using UnityEngine;
using System.Collections;

public class BrickPileController : MonoBehaviour {

    public static int width = 10;
    ArrayList myBricks;

	// Use this for initialization
	void Start () {
        myBricks = new ArrayList();
	}
	
	// Update is called once per frame
	void Update () {
	
	}

    public void AddToPile(BrickController[] bricks)
    {
        foreach(BrickController brick in bricks)
        {
            myBricks.Add(brick.gameObject);
            if (brick.transform.position.y > 4.3)
            {
                Debug.Log("x");
                UnityEditor.EditorApplication.isPlaying = false;
            }
        }
    }

    public bool CheckForGameOver()
    {
        foreach(object obj in myBricks)
        {
            GameObject brick = (GameObject)obj;
            if (brick.transform.position.y >= 1) return true;
            Debug.Log(brick.transform.position.y);
        }
        
        return false;
    }

    public int CheckAndRemove()
    {
        GameObject[] currentBricks = GetMyBricks();
        int points = 0;

        GameObject temp;
        for (int write = 0; write < currentBricks.Length; write++)
        {
            for (int sort = 0; sort < currentBricks.Length - 1; sort++)
            {
                if (currentBricks[sort].transform.position.y > currentBricks[sort + 1].transform.position.y)
                {
                    temp = currentBricks[sort + 1];
                    currentBricks[sort + 1] = currentBricks[sort];
                    currentBricks[sort] = temp;
                }
            }
        }
        //for (int write = 0; write < currentBricks.Length; write++)  Debug.Log(currentBricks[write].transform.position.y + "pos " + write);
        GameObject[] tempBricks = new GameObject[width];
        for (int i = 0; i < width; i++) tempBricks[i] = null;
        ArrayList toDelete = new ArrayList();

                for (int i = 1, counter = 0; i < currentBricks.Length; i++)
                {
                    if (Mathf.Abs(currentBricks[i].transform.position.y - currentBricks[i - 1].transform.position.y) <= 0.1)
                    {
                        if (counter == 0)
                        {
                            tempBricks[counter] = currentBricks[i - 1];
                            tempBricks[counter + 1] = currentBricks[i];
                            counter += 2;
                        }
                        else
                        {
                            tempBricks[counter] = currentBricks[i];
                            counter++;
                        }
                    }
                    else counter = 0;

                        if(counter == width)
                        {
                            //Debug.Log("Point!");
                            points += 1;
                            for(int j = 0; j< width; j++)
                            {
                                //Debug.Log("Destroy!" + j);
                                if(tempBricks[j] != null) toDelete.Add(tempBricks[j]);
                                tempBricks[j] = null;
                            }
                            counter = 0;
                        }
                        //Debug.Log("Time=" + Time.time + " i=" + i + " counter=" + counter + " ");
                }

        float maxY = -4.0f;
        foreach (Object obj in toDelete)
        {
            if(((GameObject)obj).transform.position.y > maxY)
            {
                maxY = ((GameObject)obj).transform.position.y;
            }
            Object.DestroyImmediate(obj);
            myBricks.Remove(obj);
        }

        currentBricks = GetMyBricks();

        foreach(GameObject obj in currentBricks)
        {
            if(obj.transform.position.y > maxY)
            {
                obj.transform.position = new Vector3(obj.transform.position.x, obj.transform.position.y - 
                    points*(GameController.distance), obj.transform.position.z);
            }
        }

        return points;
    }

    public GameObject[] GetMyBricks()
    {
        GameObject[] go = new GameObject[myBricks.Count];
        int i = 0;
        foreach(Object obj in myBricks)
        {
            go[i] = (GameObject)obj;
            i++;
        }
        return go;
    }
}
