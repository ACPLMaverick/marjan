using UnityEngine;
using System.Collections;

public class Player : MonoBehaviour
{
    #region Fields

    [SerializeField]
    protected Color _MyColor;
    [SerializeField]
    protected SnakeHead _MySnakeHead;

    #endregion

    #region Properties

    public Color MyColor { get { return _MyColor; } }

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

    #endregion

    #region Functions Protected

    protected void UpdateControls()
    {
        if(Input.GetKey(KeyCode.UpArrow))
        {
            _MySnakeHead.AssignDirection(SnakeHead.DirectionType.UP);
        }
        else if(Input.GetKey(KeyCode.RightArrow))
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

    #endregion
}
