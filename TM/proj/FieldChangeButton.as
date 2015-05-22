package  
{
	import away3d.core.base.Geometry;
	import away3d.materials.MaterialBase;
	import away3d.events.MouseEvent3D;
	import flash.geom.ColorTransform;
	import away3d.materials.TextureMaterial;
	
	public class FieldChangeButton extends SceneObject
	{
		public static const SIDE_LEFT:uint = 0;
		public static const SIDE_RIGHT:uint = 1;
		
		protected const COLOR_IDLE:uint = 0xFFFFFFFF;
		protected const COLOR_HOVER:uint = 0xFFFF0000;
		
		private var target:FieldViewer;
		private var mySide:uint;
		
		public function FieldChangeButton(geometry:Geometry, material:MaterialBase, target:FieldViewer, side:uint, 
										  name:String = null) 
		{
			super(geometry, material, name, true);
			
			this.target = target;
			if(side > 0)
				this.mySide = SIDE_RIGHT;
			else
				this.mySide = SIDE_LEFT;
				
			(this.material as TextureMaterial).colorTransform = new ColorTransform();
		}
		
		public override function ActionClick(me:MouseEvent3D) : void
		{
			if(target != null)
			{
				target.PauseVideoIfPlaying();
				if(mySide == SIDE_LEFT)
				{
					target.ChangeLeft();
				}
				else
				{
					target.ChangeRight();
				}
			}
		}
		
		public override function ActionHoverIn(me:MouseEvent3D) : void
		{
			(this.material as TextureMaterial).colorTransform.color = COLOR_HOVER;
		}
		
		public override function ActionHoverOut(me:MouseEvent3D) : void
		{
			(this.material as TextureMaterial).colorTransform.color = COLOR_IDLE;
		}
	}
	
}
