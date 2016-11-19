using UnityEngine;
using System.Collections.Generic;

public class SnakeBody : MonoBehaviour
{
    #region Classes

    protected class DirectionChange
    {
        public Vector2 Direction;
        public float CompletionProgress;

        public DirectionChange(Vector2 dir)
        {
            Direction = dir;
            CompletionProgress = 0.0f;
        }
    }

    #endregion

    #region Fields

    #endregion

    #region Properties

    public Player MyPlayer { get; protected set; }
    public Vector2 Direction { get; protected set; }
    public SnakeHead Head { get; protected set; }

    /// <summary>
    /// Part further away from the head.
    /// </summary>
    public SnakeBody Next { get; protected set; }

    /// <summary>
    /// Part closer to the head.
    /// </summary>
    public SnakeBody Previous { get; protected set; }

    #endregion

    #region Protected

    protected Queue<DirectionChange> _directionQueue = new Queue<DirectionChange>();
    protected Transform _transform;
    protected SpriteRenderer _spriteRenderer;
    protected BoxCollider2D _collider;
    protected Vector2 _sizeWorld;
    protected Vector2 _lastAddedDirection;
    protected float _distanceSinceLastDirectionChange = 0.0f;
    protected bool _initialized;


    #endregion

    #region MonoBehaviours

    protected virtual void Awake()
    {
        _transform = GetComponent<Transform>();
        _spriteRenderer = GetComponent<SpriteRenderer>();
        _collider = GetComponent<BoxCollider2D>();
        _sizeWorld = _spriteRenderer.bounds.extents * 2.0f;

        _initialized = false;
    }

    // Use this for initialization
    protected virtual void Start ()
    {
        
	}

    // Update is called once per frame
    protected virtual void Update ()
    {
        if(_initialized)
        {
            float scalarShift = Head.Speed * Time.deltaTime;
            _distanceSinceLastDirectionChange += scalarShift;
            _transform.position += new Vector3(Direction.x, Direction.y, 0.0f) * scalarShift;

            if(Previous != null)
            {
                if(Previous.Direction != _lastAddedDirection)
                {
                    AddDirectionChange(Previous.Direction);
                }

                UpdateDirectionChanges();
            }
        }
	}

    #endregion

    #region Functions Public

    public virtual void Initialize(Player player, SnakeHead head, SnakeBody next, SnakeBody prev, uint number)
    {
        _distanceSinceLastDirectionChange = 0.0f;
        MyPlayer = player;
        Direction = head.Direction;
        _lastAddedDirection = Direction;
        Head = head;
        Next = next;
        Previous = prev;

        Vector3 offset = -Direction;
        offset.z = 0.0f;
        for (int i = 0; i < 2; ++i)
        {
            offset[i] *= number * _sizeWorld[i];
        }

        _transform.position = Head.GetComponent<Transform>().position + offset;

        _initialized = true;
    }

    #endregion

    #region Functions Protected

    protected void AddDirectionChange(Vector2 newDirection)
    {
        _directionQueue.Enqueue(new DirectionChange(newDirection));
        _lastAddedDirection = newDirection;
    }

    protected void UpdateDirectionChanges()
    {
        if(_directionQueue.Count != 0)
        {
            DirectionChange dc = _directionQueue.Peek();
            dc.CompletionProgress += Head.Speed * Time.deltaTime;

            if (dc.CompletionProgress >= _sizeWorld.x)   // assuming size world is square
            {
                _directionQueue.Dequeue();
                Direction = dc.Direction;
                _distanceSinceLastDirectionChange = 0.0f;
                
                // fix the position as it could exceed the previous
                if (Direction == Vector2.up || Direction == Vector2.down)
                {
                    _transform.position = new Vector3(Previous.GetComponent<Transform>().position.x, _transform.position.y, _transform.position.z);
                }
                else
                {
                    _transform.position = new Vector3(_transform.position.x, Previous.GetComponent<Transform>().position.y, _transform.position.z);
                }
            }
        }
    }

    #endregion
}
