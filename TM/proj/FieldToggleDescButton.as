package  
{
	import away3d.core.base.Geometry;
	import away3d.materials.TextureMaterial;
	import flash.text.TextField;
	import flash.text.TextFormat;
	import flash.display.BitmapData;
	import away3d.textures.BitmapTexture;
	import away3d.events.MouseEvent3D;
	
	public class FieldToggleDescButton extends FieldToggleButton
	{
		protected var target:SceneObject;

		public function FieldToggleDescButton(geometry:Geometry, target:SceneObject, name:String = null) 
		{
			this.TEXT_HIDE = "Hide description";
			this.TEXT_SHOW = "Show description";
			
			super(geometry, name);
			
			if(target != null && !(target is FieldDescription))
			{
				throw new ArgumentError("Target should be FieldDescription or null.");
			}
			
			this.target = target;
		}
		
		public override function ActionClicked() : void
		{
			if(target != null)
			{
				(target as FieldDescription).Opened = true;
			}
		}
		
		public override function ActionUnclicked() : void
		{
			if(target != null)
			{
				(target as FieldDescription).Opened = false;
			}
		}
	}
	
}
