package  
{
	import away3d.core.base.Geometry;
	import away3d.materials.TextureMaterial;
	import flash.utils.Timer;
	import flash.events.TimerEvent;
	import flash.display.BitmapData;
	import away3d.textures.BitmapTexture;
	import flash.geom.Point;
	import flash.geom.Vector3D;
	import away3d.events.MouseEvent3D;
	import away3d.textures.VideoTexture;
	
	public class FieldViewer extends SceneObject
	{
		private var pics:Array;
		private var currentPicSelected:uint = 0;
		
		private var helperMesh:SceneObject;
		
		private var helperIsVisible:Boolean = false;
		public var isZoomed:Boolean = false;
		
		private var blendTimer:Timer;
		
		public function FieldViewer(geometry:Geometry, pics:Array, videos:Array, name:String = null) 
		{
			this.pics = new Array();
			
			this.blendTimer = new Timer(33, 16);
			
			var currentMat:TextureMaterial = null;
			if(pics != null && pics.length > 0)
			{
				for each(var pic:TextureMaterial in pics)
				{
					this.pics.push(pic);
				}
			}
			if(videos != null && videos.length > 0)
			{
				for each(var vid:TextureMaterial in videos)
				{
					this.pics.push(vid);
				}
			}
			
			if(this.pics.length > 0)
			{
				currentMat = this.pics[currentPicSelected] as TextureMaterial;
			}

			super(geometry, currentMat, name, true);
			
			if(this.pics != null && this.pics.length > 1)
			{
				this.helperMesh = new SceneObject(this.geometry, null, null, false);
				//this.helperMesh.z = this.helperMesh.z - 5;
				(this.helperMesh.material as TextureMaterial).alpha = 0;
				this.addChild(this.helperMesh);
			}
		}

		public function ChangeLeft() : void
		{
			if(this.pics.length < 2)
				return;
				
			currentPicSelected = (currentPicSelected - 1) % pics.length;
			if(currentPicSelected > pics.length - 1)
				currentPicSelected = pics.length - 1;
			//this.material = pics[currentPicSelected];
			
			var newMaterial:TextureMaterial = pics[currentPicSelected] as TextureMaterial;
			
			BlendMaterials(newMaterial);
		}
		
		public function ChangeRight() : void
		{
			if(this.pics.length < 2)
				return;
				
			var newMaterial:TextureMaterial = pics[(currentPicSelected = (currentPicSelected + 1) % pics.length)];
			
			BlendMaterials(newMaterial);
		}
		
		private function BlendMaterials(dest:TextureMaterial) : void
		{
			if(helperIsVisible)
			{
				helperIsVisible = false;
				this.material = dest;
				(this.material as TextureMaterial).alpha = 0;
			}
			else
			{
				helperIsVisible = true;
				helperMesh.material = dest;
				(helperMesh.material as TextureMaterial).alpha = 0;
			}
			
			blendTimer.reset();
			if(blendTimer.hasEventListener(TimerEvent.TIMER)) blendTimer.removeEventListener(TimerEvent.TIMER, BlendTimerHandler);
			blendTimer.addEventListener(TimerEvent.TIMER, BlendTimerHandler, false, 0, true);
			blendTimer.start();
		}
		
		private function BlendTimerHandler(e:TimerEvent) : void
		{
			if(helperIsVisible)
			{
				(helperMesh.material as TextureMaterial).alpha += 0.0625;
				(this.material as TextureMaterial).alpha -= 0.0625;
			}
			else
			{
				(helperMesh.material as TextureMaterial).alpha -= 0.0625;
				(this.material as TextureMaterial).alpha += 0.0625;
			}
		}
		
		public override function ActionClick(me:MouseEvent3D) : void
		{
			if(pics.length == 0)
			return;
			
			if(!isZoomed)
			{
				var nPos:Vector3D = System.getInstance().Cam.position.clone();
				var nTgt:Vector3D = System.getInstance().Cam.Target.clone();
				nPos.z += 60;
				nPos.y -= 20;
				nTgt.y -= 20;
				System.getInstance().CameraSingleMove(nPos, nTgt, 300);
				isZoomed = true;
				
				if((pics[currentPicSelected] as TextureMaterial).texture is VideoTexture)
				{
					var vt:CustomVideoTexture = (pics[currentPicSelected] as TextureMaterial).texture as CustomVideoTexture;
					if(!vt.player.playing || vt.player.paused)
					{
						vt.play();
					}
				}
			}
			else
			{
				isZoomed = false;
				System.getInstance().CameraSingleUnmove(300);
				
				if((pics[currentPicSelected] as TextureMaterial).texture is VideoTexture)
				{
					var vt:CustomVideoTexture = (pics[currentPicSelected] as TextureMaterial).texture as CustomVideoTexture;
					if(vt.player.playing)
					{
						vt.pause();
					}
				}
			}
		}
		
		public override function ActionHoldIn() : void
		{
			if(holdMe != null && hold)
			{
				var mX:Number = MouseController.getInstance().relativeMouseX;
				if(mX < 0)
				{
					PauseVideoIfPlaying();
					ChangeRight();
				}
				else if(mX > 0)
				{
					PauseVideoIfPlaying();
					ChangeLeft();
				}
				holdMe = null;
				hold = false;
			}
		}
		
		public function PauseVideoIfPlaying() : void
		{
			if((pics[currentPicSelected] as TextureMaterial).texture is CustomVideoTexture)
				{
					var vt:CustomVideoTexture = (pics[currentPicSelected] as TextureMaterial).texture as CustomVideoTexture;
					if(vt.player.playing)
					{
						vt.pause();
					}
				}
		}
		
		public function StopVideos() : void
		{
			for each(var mat:TextureMaterial in pics)
			{
				if(mat.texture is CustomVideoTexture)
				{
					if((mat.texture as VideoTexture).player.playing || (mat.texture as VideoTexture).player.paused)
					{
						(mat.texture as CustomVideoTexture).stop();
					}
				}
			}
			
		}
	}
	
}
