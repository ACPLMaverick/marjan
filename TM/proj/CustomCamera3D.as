package  
{
	import away3d.cameras.Camera3D;
	import away3d.cameras.lenses.LensBase;
	import flash.geom.Vector3D;
	import flash.events.MouseEvent;
	import away3d.core.math.Vector3DUtils;
	import flash.utils.getTimer;
	
	public class CustomCamera3D extends Camera3D
	{
		private var lerpFactor:Number = 0;
		private var lerpStep:Number = 0;
		private var activeMove:Boolean = false;
		private var oldPos:Vector3D;
		private var newPos:Vector3D;
		private var oldTgt:Vector3D;
		private var newTgt:Vector3D;
		
		private var totalScroll:Number = 0;
		private var startPos:Vector3D;
		
		private var target:Vector3D;
		
		public var locked:Boolean = true;
		private var followMouse:Boolean = false;
		
		public function CustomCamera3D(lens:LensBase = null) 
		{
			super(lens);
			
			System.getInstance().MyStage.addEventListener(MouseEvent.MOUSE_WHEEL, ScrollHandler);
			System.getInstance().MyStage.addEventListener(MouseEvent.MOUSE_DOWN, MMBDown);
			System.getInstance().MyStage.addEventListener(MouseEvent.MOUSE_UP, MMBUp);
		}
		
		public function Update() : void
		{
			if(activeMove)
			{
				if(lerpFactor <= 0)
				{
					lerpFactor = 0;
					activeMove = false;
					this.position = this.newPos;
					this.lookAt(this.newTgt);
					startPos = this.position.clone();
				}
				else
				{
					this.position = System.Vector3DLerp(newPos, this.oldPos, lerpFactor);
					this.lookAt(System.Vector3DLerp(newTgt, this.oldTgt, lerpFactor));
					lerpFactor -= lerpStep;
				}
			}
			else if(followMouse)
			{
				var mX:Number = -MouseController.getInstance().relativeMouseX;
				var mY:Number = MouseController.getInstance().relativeMouseY;
				
				var right:Vector3D = this.rightVector.clone();
				right.normalize();
				right.x *= mX * 2;
				right.y *= mX * 2;
				right.z *= mX * 2;
				
				var mup:Vector3D = new Vector3D(0, 1, 0);
				mup.normalize();
				mup.x *= mY * 2;
				mup.y *= mY * 2;
				mup.z *= mY * 2;
				
				var movement:Vector3D = right.add(mup);
				//trace(movement);
				movement = movement.add(this.position);
				
				
//				if(Math.abs(movement.y - startPos.y) < 100 &&
//				   Math.abs(movement.z - startPos.z) < 100)
//				{
//					this.position.y = movement.y;   
//					this.position.z = movement.z;
//				}
				
				if(Math.abs(movement.y - startPos.y) > 100)
					movement.y = this.position.y;
				if(Math.abs(movement.z - startPos.z) > 100)
					movement.z = this.position.z;

				this.position = movement;
			}
		}
		
		public override function lookAt(target:Vector3D, upAxis:Vector3D = null) : void
		{
			super.lookAt(target, upAxis);
			this.target = target;
		}
		
		public function SmoothMovement(newPos:Vector3D, newTgt:Vector3D, timeMS:uint) : void
		{
			this.newPos = newPos;
			this.newTgt = newTgt;
			this.oldPos = this.position.clone();
			this.oldTgt = this.target.clone();
			
			
			if(timeMS == 0)
			{
				this.position = this.newPos;
				this.lookAt(this.newTgt);
			}
			else
			{
				this.activeMove = true;
				this.lerpFactor = 1;
				this.lerpStep = 1/((timeMS as Number) / 33.33);
			}
		}
		
		private function ScrollHandler(m:MouseEvent) : void
		{
			if(!locked && !activeMove)
			{
				var scrollAmount:Number = m.delta;
				if(totalScroll + scrollAmount > 35 || totalScroll + scrollAmount < 0)
					return;
				
				var dir:Vector3D = this.target.clone().subtract(this.position);
				dir.normalize();
				dir.x *= scrollAmount * 10;
				dir.y *= scrollAmount * 10;
				dir.z *= scrollAmount * 10;
				this.position = this.position.add(dir);
				totalScroll += scrollAmount;
				//trace(totalScroll);
			}
		}
		
		private function MMBDown(m:MouseEvent) : void
		{
			if(!locked && !activeMove && !m.altKey)
			{
				this.followMouse = true;
			}
		}
		
		private function MMBUp(m:MouseEvent) : void
		{
			this.followMouse = false;
		}
		
		public function get Target() : Vector3D
		{
			return target;
		}
		
		public function get Locked() : Boolean
		{
			return locked;
		}
		
		public function set Locked(value:Boolean) : void
		{
			locked = value;
			totalScroll = 0;
			startPos = this.position.clone();
		}
	}
	
}
