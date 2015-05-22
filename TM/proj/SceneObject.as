package  
{
	import away3d.entities.Mesh;
	import away3d.core.base.Geometry;
	import away3d.materials.MaterialBase;
	import away3d.events.MouseEvent3D;
	import flash.utils.Timer;
	import flash.events.TimerEvent;

	public class SceneObject extends Mesh
	{
		protected static const TIMER_SPEED:int = 250;
		protected static const TIMER_REPEAT:int = 1;
		
		protected var timer:Timer;
		protected var justFromOut:Boolean = false;
		protected var hold:Boolean = false;
		protected var holdMe:MouseEvent3D = null;
		
		public function SceneObject(geometry:Geometry, material:MaterialBase = null, name:String = null, ifInteractive:Boolean = false) 
		{
			super(geometry, material);
			if(name != null) this.name = name;
			
			if(ifInteractive)
			{
				this.timer = new Timer(TIMER_SPEED, TIMER_REPEAT);
				EnableInteractivity();
			}
		}

		public function Update() : void
		{
			// do nothing
			if(this.numChildren > 0)
			{
				for(var i:uint = 0; i < this.numChildren; ++i)
				{
					(this.getChildAt(i) as SceneObject).Update();
				}
			}
		}
		
		public function EnableInteractivity() : void
		{
			this.mouseEnabled = true;
			
			if(this.timer == null)
				this.timer = new Timer(TIMER_SPEED, TIMER_REPEAT);
			
			this.addEventListener(MouseEvent3D.MOUSE_DOWN, MouseDownHandler);
			this.addEventListener(MouseEvent3D.MOUSE_UP, MouseUpHandler);
			this.addEventListener(MouseEvent3D.MOUSE_OVER, ActionHoverIn);
			this.addEventListener(MouseEvent3D.MOUSE_OUT, ActionHoverOut);
		}
		
		public function DisableInteractivity() : void
		{
			this.mouseEnabled = false;
			
			if(this.hasEventListener(MouseEvent3D.MOUSE_DOWN))
			   this.removeEventListener(MouseEvent3D.MOUSE_DOWN, MouseDownHandler);
			if(this.hasEventListener(MouseEvent3D.MOUSE_UP))
			   this.removeEventListener(MouseEvent3D.MOUSE_UP, MouseUpHandler);
			if(this.hasEventListener(MouseEvent3D.MOUSE_OVER))
			   this.removeEventListener(MouseEvent3D.MOUSE_OVER, ActionHoverIn);
			if(this.hasEventListener(MouseEvent3D.MOUSE_OUT))
			   this.removeEventListener(MouseEvent3D.MOUSE_OUT, ActionHoverOut);
		}
		
		public function ActionClick(me:MouseEvent3D) : void
		{
			//trace("Click! " + this.name);
		}
		
		public function ActionHoldIn() : void
		{
			if(holdMe != null && hold)
			{
				//trace("HoldIn! " + this.name);
				holdMe = null;
				hold = false;
			}
		}
		
		public function ActionHoldOut(me:MouseEvent3D) : void
		{
			//trace("HoldOut! " + this.name);
		}
		
		public function ActionHoverIn(me:MouseEvent3D) : void
		{
			//trace("HoverIn! " + this.name);
		}
		
		public function ActionHoverOut(me:MouseEvent3D) : void
		{
			//trace("HoverOut! " + this.name);
		}
		
		protected function MouseDownHandler(me:MouseEvent3D) : void
		{
			//trace("Down");
			timer.reset();
			if(timer.hasEventListener(TimerEvent.TIMER_COMPLETE)) timer.removeEventListener(TimerEvent.TIMER_COMPLETE, MouseTimerHandler);
			timer.addEventListener(TimerEvent.TIMER_COMPLETE, MouseTimerHandler, false, 0, true);
			timer.start();
			
			holdMe = me;
		}
		
		protected function MouseUpHandler(me:MouseEvent3D) : void
		{
			if(timer.running)
			{
				timer.reset();
				timer.stop();
				//trace("UpClick");
				this.ActionClick(me);
			}
			else
			{
				timer.reset();
				timer.stop();
				//trace("Up");
				this.hold = true;
				this.ActionHoldOut(me);
			}
		}
		
		protected function MouseTimerHandler(e:TimerEvent) : void
		{
			this.ActionHoldIn();
		}
	}
	
}
