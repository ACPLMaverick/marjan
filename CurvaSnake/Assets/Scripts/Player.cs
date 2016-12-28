﻿using UnityEngine;
using UnityEngine.Events;
using System.Collections.Generic;

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

    #endregion

    #region MonoBehaviours

    // Use this for initialization
    protected virtual void Start()
    {
        _MySnakeHead.GetComponent<Transform>().position = GetComponent<Transform>().position;
        _MySnakeHead.Initialize(this);
        _MySnakeHead.SnakePositionChanged.AddListener(OnPositionChanged);
    }

    // Update is called once per frame
    protected virtual void Update()
    {
    }

    #endregion

    #region Functions Public

    public virtual void Initialize(int id)
    {
        MyID = id;
    }

    public virtual void UpdateFromPlayerData(Network.PlayerData data)
    {
        // this function does nothing in default implementation
    }

    public void AddPoints(int count)
    {
        Points += count;
    }

    public void Stop()
    {
        _MySnakeHead.AssignDirection(SnakeHead.DirectionType.STOP);
    }

    public void Lose()
    {
        EventLose.Invoke(this);
    }

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

    }

    protected virtual void OnPositionChanged()
    {

    }

    #endregion
}
