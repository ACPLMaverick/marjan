using UnityEngine;
using System.Collections.Generic;

public class SnakeHead : SnakeBody
{
    #region Enum

    public enum DirectionType
    {
        UP,
        DOWN,
        STOP,
        RIGHT,
        LEFT
    }

    #endregion
    
    #region Fields

    [SerializeField]
    protected GameObject _BodyPartPrefab;
    [SerializeField]
    protected float _StartSpeed = 1.0f;
    [SerializeField]
    protected uint _StartNumberOfParts = 2;
    [SerializeField]
    protected DirectionType _StartDirection = DirectionType.UP;

    #endregion

    #region Properties

    public float Speed { get; protected set; }

    #endregion

    #region Protected

    protected List<SnakeBody> _allBodyParts = new List<SnakeBody>();
    protected DirectionType _currentDirectionType;

    #endregion

    #region MonoBehaviours

    // Use this for initialization
    protected override void Start()
    {
        base.Start();

        _distanceSinceLastDirectionChange = _sizeWorld.x;
        Speed = _StartSpeed;
        _currentDirectionType = (DirectionType)(((int)_StartDirection + 2) % 4);
        if(_currentDirectionType == DirectionType.STOP)
        {
            ++_currentDirectionType;
        }
        AssignDirection(_StartDirection);

        // this is hard coded as it cannot be the other way
        Head = this;
        Previous = null;
    }

    // Update is called once per frame
    protected override void Update()
    {
        base.Update();
    }

    protected override void OnTriggerEnter2D(Collider2D coll)
    {
        if (coll.gameObject.CompareTag("fruit"))
        {
            Fruit fr = coll.gameObject.GetComponent<Fruit>();
            if (fr != null)
            {
                PickFruit(fr);
            }
        }
        else if(coll.gameObject.CompareTag("snake") || coll.gameObject.CompareTag("head"))
        {
            // do nothing
        }
        else
        {
            Kill();
        }
    }

    #endregion

        #region Functions Public

        /// <summary>
        /// This override of the base Initialize only calls the Head-specific Initialize(Player) function. You should not use it.
        /// </summary>
    public override void Initialize(Player player, SnakeHead head, SnakeBody next, SnakeBody prev, uint number)
    {
        Initialize(player);
    }

    public void Initialize(Player player)
    {
        MyPlayer = player;
        GetComponent<SpriteRenderer>().color = MyPlayer.MyColor;

        if(_allBodyParts.Count != 0)
        {
            for(int i = 0; i < _allBodyParts.Count; ++i)
            {
                DestroyImmediate(_allBodyParts[i].gameObject, false);
            }
        }
        _allBodyParts.Clear();

        for(int i = 1; i <= _StartNumberOfParts; ++i)
        {
            _allBodyParts.Add(CreateNewPart());
        }

        for (int i = 0; i < _StartNumberOfParts; ++i)
        {
            SnakeBody next, prev;

            if(i == _StartNumberOfParts - 1)
            {
                next = null;
            }
            else
            {
                next = _allBodyParts[i + 1];
            }

            if(i == 0)
            {
                prev = this;
            }
            else
            {
                prev = _allBodyParts[i - 1];
            }

            _allBodyParts[i].Initialize(MyPlayer, this, next, prev, (uint)(i + 1));
        }

        Next = _allBodyParts[0];
        _initialized = true;
    }

    public override void Kill()
    {
        base.Kill();
        MyPlayer.Lose();
    }

    public void AssignDirection(DirectionType dir)
    {
        if(_distanceSinceLastDirectionChange >= _sizeWorld.x &&     // assuming it is square
            (Mathf.Abs((int)dir - (int)_currentDirectionType) > 1))   
        {
            _distanceSinceLastDirectionChange = 0.0f;
            _currentDirectionType = dir;
            switch (dir)
            {
                case DirectionType.DOWN:
                    Direction = Vector2.down;
                    break;
                case DirectionType.LEFT:
                    Direction = Vector2.left;
                    break;
                case DirectionType.RIGHT:
                    Direction = Vector2.right;
                    break;
                case DirectionType.UP:
                    Direction = Vector2.up;
                    break;
                default:
                    Direction = Vector2.zero;
                    break;
            }
        }
        else if(dir == DirectionType.STOP)
        {
            _distanceSinceLastDirectionChange = 0.0f;
            _currentDirectionType = dir;
            Direction = Vector2.zero;
        }
    }

    public void OnFruitCollected(float addition)
    {
        Speed += addition;
    }

    #endregion

    #region Functions Protected

    protected SnakeBody CreateNewPart()
    {
        GameObject obj = Instantiate(_BodyPartPrefab);
        obj.GetComponent<SpriteRenderer>().color = MyPlayer.MyColor;
        SnakeBody b = obj.GetComponent<SnakeBody>();
        return b;
    }

    #endregion
}
