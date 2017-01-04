using UnityEngine;
using UnityEngine.Events;
using System.Collections.Generic;

/// <summary>
/// Player's class, contains f.e. number of points, his body (SnakeBody objects) and ID
/// </summary>
public class Player : MonoBehaviour
{
    #region Fields

    [SerializeField]
    protected Color _MyColor;
    [SerializeField]
    protected SnakeHead _MySnakeHead;

    #endregion

    #region Events

    public class UnityEventPlayerLose : UnityEvent<Player> { }

    public UnityEventPlayerLose EventLose = new UnityEventPlayerLose();

    #endregion

    #region Properties

    public int MyID { get; protected set; }
    public Color MyColor { get { return _MyColor; } }
    public int Points { get; protected set; }

    #endregion

    #region Protected

    protected PlayerSpawner _AssignedSpawner;
    protected Color[] _Colors = { Color.red, Color.green, Color.yellow, Color.magenta };

    #endregion

    #region MonoBehaviours

    // Use this for initialization
    protected virtual void Start()
    {

    }

    // Update is called once per frame
    protected virtual void Update()
    {
    }

    #endregion

    #region Functions Public

    /// <summary>
    /// Player's initialize with new ID given by server
    /// </summary>
    /// <param name="id"></param>
    public virtual void Initialize(int id)
    {
        MyID = id;
        //Spawning player in spawner position --
        _AssignedSpawner = GameController.Instance.GetSpawner(MyID);
        if (_AssignedSpawner != null)
        {
            _AssignedSpawner.IsPlayerAssigned = true;
        }

        //Set color based on ID
        _MyColor = _Colors[(MyID - 1) % _Colors.Length];

        _MySnakeHead.GetComponent<Transform>().position = GetComponent<Transform>().position;
        _MySnakeHead.Initialize(this);
        _MySnakeHead.SnakePositionChanged.AddListener(OnPositionChanged);

        ChangePlayerLocationOnStart(_AssignedSpawner.MyPosition);
    }

    /// <summary>
    /// Updates player's data received from server
    /// </summary>
    /// <param name="data"></param>
    public virtual void UpdateFromPlayerData(Network.PlayerData data)
    {
        // this function does nothing in default implementation
    } 

    /// <summary>
    /// Adds points after eating apple
    /// </summary>
    /// <param name="count"></param>
    public void AddPoints(int count)
    {
        Points += count;
    }

    /// <summary>
    /// Stops player from movement
    /// </summary>
    public void Stop()
    {
        _MySnakeHead.AssignDirection(SnakeHead.DirectionType.STOP);
    }

    /// <summary>
    /// Invokes EventLose event
    /// </summary>
    public void Lose()
    {
        EventLose.Invoke(this);
    }

    /// <summary>
    /// Destroys player's game object
    /// </summary>
    public void DestroyPlayer()
    {
        //Added this because invoking Lose() didn't work
        _MySnakeHead.DestroyBody();
        Destroy(_MySnakeHead.gameObject);
        Destroy(this);
    }

    /// <summary>
    /// Sends player's data to PlayerData structure
    /// </summary>
    /// <returns></returns>
    public Network.PlayerData GetPlayerData()
    {
        Network.PlayerData data = new Network.PlayerData();

        data.PlayerID = MyID;
        data.Points = Points;
        data.PartsCount = _MySnakeHead.PartsCount + 1;

        List<SnakeBody> partsBent = _MySnakeHead.PartsBent;

        data.PartsBendsCount = partsBent.Count;
        data.PartsBentPositions = new Vector2[data.PartsBendsCount];
        data.PartsBentDirections = new SnakeHead.DirectionType[data.PartsBendsCount];

        for (int i = 0; i < data.PartsBendsCount; ++i)
        {
            data.PartsBentPositions[i] = partsBent[i].transform.position;
            data.PartsBentDirections[i] = SnakeHead.DirectionToDirectionType(partsBent[i].Direction);
        }

        data.CollisionAtPart = _MySnakeHead.LastCollisionID;

        return data;
    }

    /// <summary>
    /// Sets player's movement direction on initialize
    /// </summary>
    /// <param name="target"></param>
    public void ChangePlayerLocationOnStart(Vector3 target)
    {
        GetComponent<Transform>().position = target;
        _MySnakeHead.GetComponent<Transform>().position = target;

        for(int i = 0; i < _MySnakeHead.PartsCount; ++i)
        {
            Transform bodyPart = _MySnakeHead.GetBodyPartPosition(i);
            bodyPart.Translate(target);
        }
    } 

    #endregion

    #region Functions Protected

    protected void UpdateHead()
    {
        if (_MySnakeHead != null)
        {
            _MySnakeHead.Tick();
        }
    }

    protected void ApplyDirectlyPlayerData(Network.PlayerData data)
    {
        _MySnakeHead.SetPositionsAndDirectionsForAllParts(data.PartsCount, data.PartsBentPositions, data.PartsBentDirections);
    }

    protected virtual void OnPositionChanged()
    {
        
    }

    #endregion
}
