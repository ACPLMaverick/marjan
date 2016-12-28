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

    [SerializeField]
    protected int _PenaltyPointsPerLostPart = 1;

    #endregion

    #region Properties

    public Player MyPlayer { get; protected set; }
    public Vector2 Direction { get; protected set; }
    public SnakeHead Head { get; protected set; }

    /// <summary>
    /// Part further away from the head.
    /// </summary>
    public SnakeBody Next { get; set; }

    /// <summary>
    /// Part closer to the head.
    /// </summary>
    public SnakeBody Previous { get; set; }

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

	}

    protected virtual void OnTriggerEnter2D(Collider2D coll)
    {
        if(coll.gameObject.CompareTag("fruit"))
        {
            Fruit fr = coll.gameObject.GetComponent<Fruit>();
            if(fr != null)
            {
                PickFruit(fr);
            }
        }
        else if(coll.gameObject.CompareTag("head"))
        {
            SnakeHead head = coll.GetComponent<SnakeHead>();
            if (head != Previous)
            {
                Head.RegisterCollision(this);
                Head.BodyPartCleanup();
                Kill();
            }
        }
        else if (coll.gameObject.CompareTag("snake"))
        {
            SnakeBody body = coll.GetComponent<SnakeBody>();
            if(body != Previous && body != Next)
            {
                Head.RegisterCollision(this);
                Head.BodyPartCleanup();
                Kill();
            }
        }
        else
        {
            Head.RegisterCollision(this);
            Head.BodyPartCleanup();
            Kill();
        }
    }

    #endregion

    #region Functions Public

    public virtual void Initialize(Player player, SnakeHead head, SnakeBody next, SnakeBody prev, uint number)
    {
        _distanceSinceLastDirectionChange = 0.0f;
        MyPlayer = player;
        Direction = prev.Direction;
        _lastAddedDirection = Direction;
        Head = head;
        Next = next;
        Previous = prev;

        Vector3 offset = -Direction * _sizeWorld.x;
        offset.z = 0.0f;

        _transform.position = Previous.GetComponent<Transform>().position + offset;

        _initialized = true;
    }

    public virtual void Kill()
    {
        if(_initialized)
        {
            if (Next != null)
            {
                Next.Kill();
            }
            MyPlayer.AddPoints(-_PenaltyPointsPerLostPart);
            Destroy(gameObject);
            _initialized = false;
        }
    }

    public void Move()
    {
        if (_initialized)
        {
            if(_lastAddedDirection != Direction)
            {
                Direction = _lastAddedDirection;
            }

            ApplyMovement();

            if(Previous != null && Previous.Direction != _lastAddedDirection)
            {
                _lastAddedDirection = Previous.Direction;
            }
        }
    }

    #endregion

    #region Functions Protected

    protected void ApplyMovement()
    {
        float scalarMovement = _sizeWorld.x;
        Vector3 vectorMovement = new Vector3(Direction.x * scalarMovement, Direction.y * scalarMovement, 0.0f);
        _transform.position += vectorMovement;
    }

    protected void PickFruit(Fruit fr)
    {
        Head.OnFruitCollected(fr.SpeedAddition);
        MyPlayer.AddPoints(fr.Points);
    }

    #endregion
}
