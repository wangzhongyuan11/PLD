// Keep download/decoding errors distinct from WeChat and iOS authorization.
function call(name, options) {
  return new Promise(function (resolve, reject) {
    wx[name](Object.assign({}, options, { success: resolve, fail: reject }));
  });
}

function permission(system, allowSettings) {
  var name = system ? "openAppAuthorizeSetting" : "openSetting";
  return call("showModal", {
    title: system ? "微信的系统相册权限未开启" : "需要相册权限",
    content: system ? "请在 iPhone 设置中允许微信添加照片，返回后继续保存。" : "请允许本小程序保存到相册，返回后继续保存。",
    confirmText: "前往设置",
  }).then(function (r) {
    if (!r.confirm || !allowSettings || !wx[name]) throw new Error("相册权限未开启，图片未保存");
    return call(name, {});
  }).then(function () { return authorize(false); });
}

function authorize(allowSettings) {
  var system = wx.getAppAuthorizeSetting && wx.getAppAuthorizeSetting();
  if (system && system.albumAuthorized === "denied") {
    if (!allowSettings) return Promise.reject(new Error("系统相册权限仍未开启"));
    return permission(true, allowSettings);
  }
  return call("getSetting", {}).then(function (r) {
    var state = (r.authSetting || {})["scope.writePhotosAlbum"];
    if (state === true) return;
    if (state === false) {
      if (!allowSettings) throw new Error("小程序相册权限仍未开启");
      return permission(false, allowSettings);
    }
    return call("authorize", { scope: "scope.writePhotosAlbum" });
  });
}

function saveImage(fileID) {
  var local, stage = "下载图片", info;
  return new Promise(function (resolve, reject) {
    wx.cloud.downloadFile({ fileID: fileID, success: resolve, fail: reject });
  }).then(function (r) {
    stage = "检查图片";
    return call("getImageInfo", { src: r.tempFilePath }).then(function (value) {
      info = value;
      var type = (info.type || "").toLowerCase();
      if (["png", "jpeg", "jpg"].indexOf(type) < 0) throw new Error("图片不是可保存的 PNG/JPEG");
      // Some preview URLs end in .png while returning JPEG. Give iOS an
      // extension matching the decoded bytes, without recompressing the image.
      local = wx.env.USER_DATA_PATH + "/pld_save_" + Date.now() + "." + (type === "png" ? "png" : "jpg");
      return new Promise(function (resolve, reject) {
        wx.getFileSystemManager().copyFile({ srcPath: r.tempFilePath, destPath: local, success: resolve, fail: reject });
      });
    });
  }).then(function () {
    stage = "相册授权";
    return authorize(true);
  }).then(function () {
    stage = "保存到相册";
    return call("saveImageToPhotosAlbum", { filePath: local });
  }).catch(function (err) {
    var raw = (err && (err.errMsg || err.message)) || "未知错误";
    console.error("PLD 图片保存失败", { stage: stage, errMsg: raw, errno: err && err.errno,
      type: info && info.type, width: info && info.width, height: info && info.height });
    throw new Error(stage + "失败：" + raw + (err && err.errno != null ? "（" + err.errno + "）" : ""));
  }).finally(function () {
    if (local) wx.getFileSystemManager().unlink({ filePath: local, fail: function () {} });
  });
}

module.exports = { saveImage: saveImage };
