package  
{
	import away3d.core.base.Geometry;
	import away3d.materials.TextureMaterial;
	import away3d.events.MouseEvent3D;
	
	public class FieldToggleModelButton extends FieldToggleButton
	{
		private var mParent:InfoPlane;
		
		public function FieldToggleModelButton(geometry:Geometry, par:InfoPlane, name:String = null) 
		{
			this.TEXT_HIDE = "Hide model";
			this.TEXT_SHOW = "Show model";
			
			super(geometry, name);
			
			this.mParent = par;
		}
		
		public override function ActionClicked() : void
		{
			if(mParent != null)
			{
				mParent.ShowModel();
			}
		}
	}
	
}
