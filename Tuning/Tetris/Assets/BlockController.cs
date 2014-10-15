using UnityEngine;
using System.Collections;

public class BlockController : MonoBehaviour {

	// Use this for initialization
	void Start () {
        //this.GetComponentInChildren<BrickController>().renderer.material.color = Color.red;
        //GameObject current = GameObject.Find(this.ToString());
        //Debug.Log(current);
        //foreach (var child in this.GetComponentsInChildren<BrickController>())
        //{
        //    //Debug.Log(child);
        //    child.renderer.material.color = Color.red;
        //}
	}
	
	// Update is called once per frame
	void Update () {
	    
	}

    public void BlockRotate(Vector3 rotation)
    {
        BrickController[] myBricks = GetMyBricks();

        Vector3 backwardRotation = new Vector3(rotation.x, rotation.y, -rotation.z);
        Quaternion newRotationBlock = transform.rotation;
        newRotationBlock = Quaternion.Euler(newRotationBlock.eulerAngles + rotation);

        transform.rotation = newRotationBlock;

        foreach (BrickController brick in myBricks)
        {
            brick.transform.rotation = Quaternion.Euler(brick.transform.rotation.eulerAngles + backwardRotation);
        }
    }

    public BrickController[] GetMyBricks()
    {
        return GetComponentsInChildren<BrickController>();
    }

    public ArrayList GetMyBricksAsList()
    {
        BrickController[] br = GetComponentsInChildren<BrickController>();
        ArrayList toreturn = new ArrayList();
        foreach(BrickController brick in br)
        {
            toreturn.Add(brick);
        }
        return toreturn;
    }
}
