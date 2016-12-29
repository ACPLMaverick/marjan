using UnityEngine;
using System.Collections;

public class PlayerLocal : Player
{
    #region Protected

    /** 
     * Every local player has his a reference to network client. 
     * It is responsible only for SENDING player data to server and maintaining connection.
     */
    protected Network.Client _gameClient;

    #endregion

    #region MonoBehaviours

    // Use this for initialization
    protected override void Start ()
    {
        base.Start();
	}
	
	// Update is called once per frame
	protected override void Update ()
    {
        base.Update();

        UpdateControls();
        UpdateHead();
	}

    #endregion

    #region Functions Public

    public void Initialize(int id, Network.Client clientRef)
    {
        MyID = id;
        _gameClient = clientRef;
    }

    #endregion

    #region Functions Protected

    protected void UpdateControls()
    {
        if (_MySnakeHead != null)
        {
            if (Input.GetKey(KeyCode.UpArrow))
            {
                _MySnakeHead.AssignDirection(SnakeHead.DirectionType.UP);
            }
            else if (Input.GetKey(KeyCode.RightArrow))
            {
                _MySnakeHead.AssignDirection(SnakeHead.DirectionType.RIGHT);
            }
            else if (Input.GetKey(KeyCode.DownArrow))
            {
                _MySnakeHead.AssignDirection(SnakeHead.DirectionType.DOWN);
            }
            else if (Input.GetKey(KeyCode.LeftArrow))
            {
                _MySnakeHead.AssignDirection(SnakeHead.DirectionType.LEFT);
            }
        }
    }

    protected override void OnPositionChanged()
    {
        base.OnPositionChanged();

        if(_gameClient != null)
        {
            _gameClient.SendDataToServer(GetPlayerData());
        }
    }

    #endregion
}
