using UnityEngine;
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

    public Color MyColor { get { return _MyColor; } }
    public int Points { get; protected set; }

    #endregion

    #region Protected

    #endregion

    #region MonoBehaviours

    // Use this for initialization
    void Start()
    {
        _MySnakeHead.GetComponent<Transform>().position = GetComponent<Transform>().position;
        _MySnakeHead.Initialize(this);
    }

    // Update is called once per frame
    void Update()
    {
        UpdateControls();
    }

    #endregion

    #region Functions Public

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

    #endregion

    #region Functions Protected

    protected void UpdateControls()
    {
        if(_MySnakeHead != null)
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

    #endregion
}
