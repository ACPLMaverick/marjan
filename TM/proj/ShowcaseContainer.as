package  
{
	import away3d.events.MouseEvent3D;
	import flash.utils.Timer;
	import flash.events.TimerEvent;
	import away3d.materials.TextureMaterial;
	import flash.events.MouseEvent;
	
	public class ShowcaseContainer extends SceneObject
	{
		private var mVisible:Boolean = true;
		private var blendTimer:Timer;
		private var rotatedByMouse:Boolean = false;
		
		public function ShowcaseContainer(name:String = null) 
		{
			super(null, null, name, true);
			blendTimer = new Timer(33, 16);
		}
		
		public override function Update() : void
		{
			if(rotatedByMouse)
			{
				var mX:Number = MouseController.getInstance().relativeMouseX;
				var mY:Number = MouseController.getInstance().relativeMouseY;

				this.rotationY -= mX / 4.0;
				this.rotationZ += mY / 4.0;
			}
		}
		
		private function ActionDown(me:MouseEvent) : void
		{
			if(me.altKey)
			{
				rotatedByMouse = true;
				holdMe = null;
			}
		}
		
		private function ActionUp(me:MouseEvent) : void
		{
			rotatedByMouse = false;
		}
		
		public override function get visible() : Boolean
		{
			return mVisible;
		}
		
		public override function set visible(value:Boolean) : void
		{
			if(mVisible != value)
			{
				blendTimer.reset();
				if(blendTimer.hasEventListener(TimerEvent.TIMER)) blendTimer.removeEventListener(TimerEvent.TIMER, BlendTimerHandler);
				blendTimer.addEventListener(TimerEvent.TIMER, BlendTimerHandler, false, 0, true);
				blendTimer.start();
				
				if(value)
				{
					this.EnableInteractivity();
					for(var i:uint = 0; i < this.numChildren; ++i)
					{
						(getChildAt(i) as SceneObject).EnableInteractivity();
					}
					
					this.rotationX = 0;
					this.rotationY = 0;
					this.rotationZ = 0;
					
					System.getInstance().MyStage.addEventListener(MouseEvent.MOUSE_DOWN, ActionDown, false, 0, true);
					System.getInstance().MyStage.addEventListener(MouseEvent.MOUSE_UP, ActionUp, false, 0, true);
				}
				else
				{
					this.DisableInteractivity();
					for(var i:uint = 0; i < this.numChildren; ++i)
					{
						(getChildAt(i) as SceneObject).DisableInteractivity();
					}
					
					System.getInstance().MyStage.removeEventListener(MouseEvent.MOUSE_DOWN, ActionDown);
					System.getInstance().MyStage.removeEventListener(MouseEvent.MOUSE_UP, ActionUp);
				}
			}
			
			mVisible = value;
		}
		
		private function BlendTimerHandler(e:TimerEvent) : void
		{
			if(mVisible)
			{
				for(var i:uint = 0; i < this.numChildren; ++i)
				{
					var tm:TextureMaterial = ((this.getChildAt(i) as SceneObject).material as TextureMaterial);
					if(tm != null)
					{
						tm.alpha = Math.min(tm.alpha + 0.0625, 1);
					}
				}
			}
			else
			{
				for(var i:uint = 0; i < this.numChildren; ++i)
				{
					var tm:TextureMaterial = ((this.getChildAt(i) as SceneObject).material as TextureMaterial);
					if(tm != null)
					{
						tm.alpha = Math.max(tm.alpha - 0.0625, 0);
					}
				}
			}
		}
	}
	
}
