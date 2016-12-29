using UnityEngine;
using System.Collections;

public class PlayerSpawner : MonoBehaviour {

    #region Fields

    [SerializeField]
    protected int _MyID;
    protected bool _IsPlayerAssigned;

    #endregion

    #region Properties

    public Vector3 MyPosition
    {
        get { return GetComponent<Transform>().position; }
        protected set { }
    }

    public int MyID { get; protected set; }

    public bool IsPlayerAssigned
    {
        get { return _IsPlayerAssigned; }
        set { _IsPlayerAssigned = value; }
    }

    #endregion

    #region MonoBehaviours

    // Use this for initialization
    void Start () {
	    
	}
	
	// Update is called once per frame
	void Update () {
	
	}

    #endregion
}
