package  
{
	import away3d.textures.BitmapTexture;
	import away3d.core.base.Geometry;
	import away3d.materials.TextureMaterial;
	import flash.text.TextField;
	import flash.display.BitmapData;
	import flash.text.TextFormat;
	
	public class FieldTitle extends SceneObject
	{
		private var text:String;
		private var texture:BitmapTexture;
		
		public function FieldTitle(geometry:Geometry, name:String = null, text:String = "SomeTitle") 
		{
			this.text = text;
			super(geometry, GenerateTitle(), name, false);
		}
		
		private function GenerateTitle() : TextureMaterial
		{
			var tempFormat:TextFormat = new TextFormat();
			tempFormat.font = "Motorwerk";
			tempFormat.align = "center";
			tempFormat.size = 225;
			tempFormat.leftMargin = 10;
			tempFormat.rightMargin = 10;
			tempFormat.color = 0xFFFFFFFF;
			var tempTF:TextField = new TextField();
			tempTF.defaultTextFormat = tempFormat;
			tempTF.text = this.text;
			tempTF.width = 2048;
			tempTF.height = 256;
			var bm:BitmapData = new BitmapData(tempTF.width, tempTF.height, true, 0);
			bm.draw(tempTF);
			this.texture = new BitmapTexture(bm, false);
			
			var mat:TextureMaterial = new TextureMaterial(this.texture, false, false , false);
			mat.alphaPremultiplied = false;
			mat.alphaBlending = true;
			mat.alphaThreshold = 0.01;
			return mat;
		}
	}
	
}
