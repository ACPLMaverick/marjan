package  
{
	import away3d.core.base.Geometry;
	import away3d.materials.TextureMaterial;
	import flash.text.TextField;
	import flash.text.TextFormat;
	import flash.display.BitmapData;
	import away3d.textures.BitmapTexture;
	import away3d.events.MouseEvent3D;
	import flash.geom.ColorTransform;
	
	public class FieldToggleButton extends SceneObject
	{
		public var TEXT_SHOW:String = null;
		public var TEXT_HIDE:String = null;
		
		protected const COLOR_IDLE:uint = 0xFFFFFFFF;
		protected const COLOR_HOVER:uint = 0xFFFF0000;
		
		protected var showMat:TextureMaterial;
		protected var hideMat:TextureMaterial;
		
		protected var mColorTransform:ColorTransform;
		
		protected var show:Boolean = true;

		public function FieldToggleButton(geometry:Geometry, name:String = null) 
		{
			if(TEXT_SHOW == null)
			{
				TEXT_SHOW = "Default show";
			}
			if(TEXT_HIDE == null)
			{
				TEXT_HIDE = "Default hide";
			}
			
			GenerateMats();
			
			var material:TextureMaterial = showMat;
			super(geometry, material, name, true);
			this.mColorTransform = new ColorTransform();
			this.showMat.colorTransform = this.mColorTransform;
			this.hideMat.colorTransform = this.mColorTransform;
		}
		
		public function GenerateMats() : void
		{
			if(showMat == null || hideMat == null)
			{
				var tempFormat:TextFormat = new TextFormat();
				tempFormat.font = "TW Cen MT Condensed";
				tempFormat.align = "center";
				tempFormat.size = 215;
				tempFormat.leftMargin = 10;
				tempFormat.rightMargin = 10;
				tempFormat.color = COLOR_IDLE;
				
				var tempTF:TextField = new TextField();
				tempTF.defaultTextFormat = tempFormat;
				tempTF.text = TEXT_SHOW;
				tempTF.width = 2048;
				tempTF.height = 256;
				var bm:BitmapData = new BitmapData(tempTF.width, tempTF.height, true, 0);
				bm.draw(tempTF);
				var tex:BitmapTexture = new BitmapTexture(bm, false);
				
				showMat = new TextureMaterial(tex, false, false, false);
				showMat.alphaPremultiplied = false;
				showMat.alphaBlending = true;
				showMat.alphaThreshold = 0.01;
				
				var tempTFh:TextField = new TextField();
				tempTFh.defaultTextFormat = tempFormat;
				tempTFh.text = TEXT_HIDE;
				tempTFh.width = 2048;
				tempTFh.height = 256;
				var bmh:BitmapData = new BitmapData(tempTFh.width, tempTFh.height, true, 0);
				bmh.draw(tempTFh);
				var texh:BitmapTexture = new BitmapTexture(bmh, false);
				
				hideMat = new TextureMaterial(texh, false, false, false);
				hideMat.alphaPremultiplied = false;
				hideMat.alphaBlending = true;
				hideMat.alphaThreshold = 0.01;
			}
		}
		
		public override function ActionClick(me:MouseEvent3D) : void
		{
				if(show)
				{
					this.show = false;
					this.material = hideMat;
					ActionClicked();
				}
				else
				{
					this.show = true;
					this.material = showMat;
					ActionUnclicked();
				}   
		}
		
		public function ActionClicked() : void
		{
			
		}
		
		public function ActionUnclicked() : void
		{
			
		}
		
		public override function ActionHoverIn(me:MouseEvent3D) : void
		{
			mColorTransform.color = COLOR_HOVER;
		}
		
		public override function ActionHoverOut(me:MouseEvent3D) : void
		{
			mColorTransform.color = COLOR_IDLE;
		}
	}
	
}
